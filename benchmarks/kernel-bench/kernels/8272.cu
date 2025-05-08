#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Vectorized load helper function
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

// Vectorized store helper function
__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

__global__ void conv1d_forward_kernel_vectorized(
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
    // Each thread processes 4 output positions when possible
    const int tid = threadIdx.x;
    const int out_ch = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int base_out_pos = (blockIdx.y * blockDim.x + tid) * 4;
    
    // Early exit if this thread is not needed
    if (base_out_pos >= L_out) return;

    // Calculate group information
    const int group_size_out = C_out / groups;
    const int group_size_in = C_in / groups;
    const int group_idx = out_ch / group_size_out;
    
    // Pre-compute input channel offset for this group
    const int in_ch_offset = group_idx * group_size_in;
    
    // Initialize output values
    float4 output = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process input channels
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        const int in_ch = in_ch_offset + local_in_ch;
        const float* weight_ptr = &w[out_ch * (group_size_in * K) + local_in_ch * K];
        
        // Pre-load weights into registers
        float weight_cache[32];  // Assuming K <= 32, adjust if needed
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            weight_cache[k] = weight_ptr[k];
        }
        
        // Process 4 output positions
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const float w_val = weight_cache[k];
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (base_out_pos + i < L_out) {
                    const int in_pos = (base_out_pos + i) * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        const float x_val = x[batch_idx * (C_in * L_in) + in_ch * L_in + in_pos];
                        float* out_ptr = reinterpret_cast<float*>(&output);
                        out_ptr[i] += x_val * w_val;
                    }
                }
            }
        }
    }
    
    // Add bias if present
    if (bias_ptr) {
        const float bias_val = bias_ptr[out_ch];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float* out_ptr = reinterpret_cast<float*>(&output);
            out_ptr[i] += bias_val;
        }
    }
    
    // Store results
    const int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + base_out_pos;
    const int remaining = min(4, L_out - base_out_pos);
    
    // Handle aligned stores when possible
    if (remaining == 4 && (reinterpret_cast<uintptr_t>(&y[out_offset]) & 15) == 0) {
        store_float4(&y[out_offset], output);
    } else {
        // Handle unaligned or partial stores
        float* out_ptr = reinterpret_cast<float*>(&output);
        for (int i = 0; i < remaining; ++i) {
            y[out_offset + i] = out_ptr[i];
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

    // Adjust block size to handle 4 elements per thread
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out,
        (L_out + (threads_per_block * 4) - 1) / (threads_per_block * 4),
        N
    );

    conv1d_forward_kernel_vectorized<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_vectorized failed: ", cudaGetErrorString(err));

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
        "Vectorized 1D Convolution forward (CUDA)"
    );
}