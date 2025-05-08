#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

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
    const unsigned WARP_SIZE = 32;
    const unsigned lane_id = threadIdx.x % WARP_SIZE;
    const unsigned warp_id = threadIdx.x / WARP_SIZE;
    
    const int n = blockIdx.x;
    const int out_ch = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_ch >= C_out) return;

    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    
    for (int base_out_pos = warp_id * WARP_SIZE; base_out_pos < L_out; base_out_pos += (blockDim.x / WARP_SIZE) * WARP_SIZE) {
        const int out_pos = base_out_pos + lane_id;
        
        if (out_pos < L_out) {
            float sum = 0.0f;
            
            for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
                const int in_ch = group_idx * group_size_in + local_in_ch;
                
                for (int k = 0; k < K; k++) {
                    const float w_val = __shfl_sync(0xffffffff, 
                        w[out_ch * (group_size_in * K) + local_in_ch * K + k], 
                        0);
                    
                    const int in_pos = out_pos * stride + k * dilation - padding;
                    float x_val = 0.0f;
                    
                    if (in_pos >= 0 && in_pos < L_in) {
                        x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                    }
                    
                    sum += x_val * w_val;
                }
            }
            
            if (bias_ptr) {
                sum += bias_ptr[out_ch];
            }
            
            if (out_pos < L_out) {
                y[n * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
            }
        }
    }
}

at::Tensor conv1d_forward_impl_warp(
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

    const int WARP_SIZE = 32;
const int TILE_SIZE = 64; // Align tile dimensions to warp boundaries
    const int WARPS_PER_BLOCK = 4;
    const int CHANNELS_PER_BLOCK = 4;
    
    dim3 block(WARP_SIZE * WARPS_PER_BLOCK, CHANNELS_PER_BLOCK);
    dim3 grid(N, (C_out + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);

    conv1d_forward_kernel_warp<<<grid, block>>>(
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
            return conv1d_forward_impl_warp(x, weight, bias, stride, padding, dilation, groups);
        },
        "Warp-optimized 1D Convolution forward (CUDA)"
    );
}