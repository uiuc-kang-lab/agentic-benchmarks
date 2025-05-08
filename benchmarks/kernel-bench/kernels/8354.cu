#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

__global__ void conv1d_forward_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
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
    const int L_out,
    const int group_size_in,
    const int group_size_out
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[blockDim.x + 2 * padding];

    const int tid = threadIdx.x;
    const int out_pos = blockIdx.x * blockDim.x + tid;
    const int out_ch = blockIdx.y;
    const int n = blockIdx.z;
    
    if (n >= N || out_ch >= C_out) return;

    const int group_idx = out_ch / group_size_out;
    float val = 0.0f;

    // Load input data into shared memory
    const int input_start = out_pos * stride - padding;
    if (tid < blockDim.x) {
        if (input_start + tid >= 0 && input_start + tid < L_in) {
            shared_input[tid] = __ldg(&x[n * C_in * L_in + group_idx * group_size_in * L_in + input_start + tid]);
        } else {
            shared_input[tid] = 0.0f;
        }
    }

    // Load weights into shared memory
    if (tid < K) {
        shared_weight[tid] = __ldg(&w[out_ch * K + tid]);
    }
    __syncthreads();

    if (out_pos < L_out) {
        #pragma unroll 4
        for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
            const int in_ch = group_idx * group_size_in + local_in_ch;
            
            #pragma unroll 4
            for (int k = 0; k < K; ++k) {
                const int in_pos = tid + k * dilation;
                if (in_pos >= 0 && in_pos < blockDim.x) {
                    val += shared_input[in_pos] * shared_weight[k];
                }
            }
        }

        if (bias) {
            val += __ldg(&bias[out_ch]);
        }

        y[n * C_out * L_out + out_ch * L_out + out_pos] = val;
    }
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
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int C_out = weight.size(0);
    const int K = weight.size(2);

    const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Invalid output length");

    auto y = torch::empty({N, C_out, L_out}, x.options());
    const float* bias_ptr = bias_opt.has_value() ? bias_opt->data_ptr<float>() : nullptr;

    const int group_size_in = C_in / groups;
    const int group_size_out = C_out / groups;

    const int block_size = 128;
    dim3 block(block_size);
    dim3 grid(
        (L_out + block_size - 1) / block_size,
        C_out,
        N
    );

    const int shared_mem_size = (block_size + 2 * padding + K) * sizeof(float);

    conv1d_forward_kernel_shared<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        stride, padding, dilation, groups, L_out,
        group_size_in, group_size_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
        [](at::Tensor x, at::Tensor weight, py::object bias,
           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) {
            return conv1d_forward_impl(x, weight,
                bias.is_none() ? c10::nullopt : c10::optional<at::Tensor>(bias.cast<at::Tensor>()),
                stride, padding, dilation, groups);
        }, "Optimized 1D Conv with shared memory"
    );
}