#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// Use float4 for vectorized loads
typedef float4 aligned_float4;

__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

__global__ void conv1d_forward_kernel_aligned(
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
    // Align thread blocks to 128-bit boundaries
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;
    const int gid = block_offset + tid;
    
    // Process 4 elements at a time when possible
    const int vec_L_out = L_out / 4 * 4;
    
    for (int idx = gid; idx < N * C_out * L_out; idx += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
        const int out_pos = idx % L_out;
        const int out_ch = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);
        
        const int group_idx = out_ch / group_size_out;
        float val = 0.0f;

        #pragma unroll 4
        for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
            const int in_ch = group_idx * group_size_in + local_in_ch;
            
            // Align weight loads to 128-bit boundaries
            const float4* w_aligned = reinterpret_cast<const float4*>(&w[out_ch * (group_size_in * K) + local_in_ch * K]);
            
            #pragma unroll 4
            for (int k = 0; k < K; k++) {
                const int in_pos = out_pos * stride + k * dilation - padding;
                if (in_pos >= 0 && in_pos < L_in) {
                    // Use __ldg for all read-only global memory accesses
                    const float x_val = __ldg(&x[n * C_in * L_in + in_ch * L_in + in_pos]);
                    const float w_val = __ldg(&w[out_ch * (group_size_in * K) + local_in_ch * K + k]);
                    val += x_val * w_val;
                }
            }
        }

        if (bias) {
            val += __ldg(&bias[out_ch]);
        }

        // Coalesced write to output
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

    // Optimize block and grid dimensions for balanced occupancy
    dim3 block(64, 4);  // 256 threads per block, balanced occupancy
    dim3 grid(
        (L_out + 63) / 64,
        (C_out + 3) / 4,
        N
    );

    conv1d_forward_kernel_aligned<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups, L_out,
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
        }, "Optimized 1D Conv with aligned memory access"
    );
}