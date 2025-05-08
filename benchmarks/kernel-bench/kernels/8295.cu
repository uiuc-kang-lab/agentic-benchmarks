#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

#define TILE_SIZE 32

__global__ void conv1d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr,
    float* __restrict__ y,
    const int N,
    const int C_in,
    const int L_in,
    const int C_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out
) {
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int out_pos = idx % L_out;
    int out_ch = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    int group_size_out = C_out / groups;
    int group_size_in = C_in / groups;
    int group_idx = out_ch / group_size_out;

    float val = 0.0f;

    // Load input tile into shared memory
    int tile_start = (out_pos * stride - padding) / TILE_SIZE * TILE_SIZE;
    int tile_end = tile_start + TILE_SIZE + 2 * padding;

    if (threadIdx.x < TILE_SIZE + 2 * padding) {
        int load_idx = tile_start + threadIdx.x;
        if (load_idx >= 0 && load_idx < L_in) {
            shared_input[threadIdx.x] = x[n * (C_in * L_in) + group_idx * group_size_in * L_in + load_idx];
        } else {
            shared_input[threadIdx.x] = 0.0f;
        }
    }

    // Load weights into shared memory
    if (threadIdx.x < K * group_size_in) {
        shared_weight[threadIdx.x] = w[out_ch * (group_size_in * K) + threadIdx.x];
    }

    __syncthreads();

    // Compute convolution using shared memory
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding - tile_start;
            if (in_pos >= 0 && in_pos < TILE_SIZE + 2 * padding) {
                val += shared_input[in_pos] * 
                       shared_weight[local_in_ch * K + k];
            }
        }
    }

    if (bias_ptr) {
        val += bias_ptr[out_ch];
    }

    y[idx] = val;
}

at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    int total_threads = N * C_out * L_out;
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;

    int shared_mem_size = (TILE_SIZE + 2 * padding + K * (C_in / groups)) * sizeof(float);

    conv1d_forward_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel failed: ", cudaGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor weight,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, weight, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CUDA) with shared memory optimization"
    );
}