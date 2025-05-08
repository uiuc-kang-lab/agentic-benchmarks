#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
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
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int out_ch = blockIdx.x;
    const int batch_idx = blockIdx.z;
    
    // Calculate group information
    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    
    // Each warp processes multiple output positions
    const int positions_per_warp = 4;
    const int warp_output_offset = (blockIdx.y * warps_per_block + warp_id) * positions_per_warp;
    
    // Register cache for frequently accessed weights
    float weight_cache[8];  // Adjust size based on typical K value
    const int weights_per_thread = (K + warp_size - 1) / warp_size;
    
    #pragma unroll
    for (int i = 0; i < weights_per_thread; i++) {
        const int w_idx = lane_id + i * warp_size;
        if (w_idx < K) {
            weight_cache[i] = w[out_ch * (group_size_in * K) + w_idx];
        }
    }
    
    // Process output positions assigned to this warp
    #pragma unroll
    for (int pos_offset = 0; pos_offset < positions_per_warp; pos_offset++) {
        const int out_pos = warp_output_offset + pos_offset;
        if (out_pos >= L_out) continue;
        
        float sum = 0.0f;
        
        // Process input channels
        for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
            const int in_ch = group_idx * group_size_in + local_in_ch;
            
            // Each thread in warp processes different kernel positions
            float thread_sum = 0.0f;
            #pragma unroll
            for (int k_offset = 0; k_offset < weights_per_thread; k_offset++) {
                const int k = lane_id + k_offset * warp_size;
                if (k < K) {
                    const int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        const float x_val = x[batch_idx * (C_in * L_in) + in_ch * L_in + in_pos];
                        thread_sum += x_val * weight_cache[k_offset];
                    }
                }
            }
            
            // Reduce within warp
            sum += warp_reduce_sum(thread_sum);
        }
        
        // First thread in warp writes result
        if (lane_id == 0) {
            if (bias_ptr) {
                sum += bias_ptr[out_ch];
            }
            y[batch_idx * (C_out * L_out) + out_ch * L_out + out_pos] = sum;
        }
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

    // Configure launch parameters for warp-based execution
    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * 32;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out,
        (L_out + warps_per_block * 4 - 1) / (warps_per_block * 4),
        N
    );

    conv1d_forward_kernel_warp<<<grid_dim, block_dim>>>(
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
        "Fast warp-based 1D Convolution forward (CUDA)"
    );
}