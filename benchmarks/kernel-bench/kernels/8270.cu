#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Warp size constant
constexpr int WARP_SIZE = 32;

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void conv1d_forward_kernel_warp(
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
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Calculate output position
    const int out_ch = blockIdx.x;
    const int out_pos = blockIdx.y * blockDim.x + tid;
    const int n = blockIdx.z;
    
    if (out_pos >= L_out) return;

    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;

    // Each thread processes a subset of input channels
    const int channels_per_thread = (group_size_in + WARP_SIZE - 1) / WARP_SIZE;
    float sum = 0.0f;

    #pragma unroll 2
    for (int c = 0; c < channels_per_thread; c++) {
        const int local_in_ch = lane_id + c * WARP_SIZE;
        if (local_in_ch < group_size_in) {
            const int in_ch = group_idx * group_size_in + local_in_ch;
            
            #pragma unroll 4
            for (int k = 0; k < K; k++) {
                const int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    const float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    const float w_val = w[out_ch * (group_size_in * K) + local_in_ch * K + k];
                    sum += x_val * w_val;
                }
            }
        }
    }

    // Perform warp-level reduction
    sum = warp_reduce_sum(sum);

    // First thread in warp writes result
    if (lane_id == 0 && out_pos < L_out) {
        if (bias_ptr) {
            sum += bias_ptr[out_ch];
        }
        y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
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
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    const int block_size = 256;
    dim3 blockSize(block_size);
    dim3 gridSize(C_out, (L_out + block_size - 1) / block_size, N);

    conv1d_forward_kernel_warp<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_warp failed: ", cudaGetErrorString(err));

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
        "Warp-optimized 1D Convolution forward (CUDA)"
    );
}