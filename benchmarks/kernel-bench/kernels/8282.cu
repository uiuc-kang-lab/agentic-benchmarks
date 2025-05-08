#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <stdint.h>

namespace py = pybind11;

// Helper functions for vectorized (128-bit) loads/stores
__device__ __forceinline__ float4 load_float4(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__device__ __forceinline__ void store_float4(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

// Kernel implementing 1D convolution using __ldg() for read-only global memory accesses
// and aligning global memory loads/stores to 128-bit boundaries
__global__ void conv1d_forward_kernel_aligned_ldg(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // can be null if no bias
    float* __restrict__ y,
    const int N,      // batch size
    const int C_in,   // input channels
    const int L_in,   // input length
    const int C_out,  // output channels
    const int K,      // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out   // output length
) {
    // Each thread processes 4 output positions when possible
    const int tid = threadIdx.x;
    const int out_ch = blockIdx.x;        // each block in x-dim corresponds to one output channel
    const int batch_idx = blockIdx.z;       // batch index
    // blockIdx.y covers groups of output positions (each thread processes 4 outputs)
    const int base_group = blockIdx.y;
    const int base_out_pos = (base_group * blockDim.x + tid) * 4;

    if (base_out_pos >= L_out) return;

    // Determine group indices
    const int group_size_out = C_out / groups;  
    const int group_size_in  = C_in / groups;   
    const int group_idx      = out_ch / group_size_out;

    // Accumulator for 4 outputs (vectorized)
    float4 output = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float* acc = reinterpret_cast<float*>(&output);

    // Iterate over the input channels in the current group
    for (int local_in_ch = 0; local_in_ch < group_size_in; ++local_in_ch) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        // Pointer to weights for this output channel and input channel
        const float* weight_ptr = &w[out_ch * (group_size_in * K) + local_in_ch * K];
        
        // Loop over kernel positions
        for (int k = 0; k < K; ++k) {
            // Use __ldg() to load weight value from global memory
            float w_val = __ldg(&weight_ptr[k]);
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int out_pos = base_out_pos + i;
                if (out_pos < L_out) {
                    int in_pos = out_pos * stride + k * dilation - padding;
                    if (in_pos >= 0 && in_pos < L_in) {
                        int idx = batch_idx * (C_in * L_in) + in_ch * L_in + in_pos;
                        // Use __ldg() for read-only load from x
                        float x_val = __ldg(&x[idx]);
                        acc[i] += x_val * w_val;
                    }
                }
            }
        }
    }

    // Add bias if available
    if (bias_ptr) {
        float b = __ldg(&bias_ptr[out_ch]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += b;
        }
    }

    // Write results back to y using vectorized store if 16-byte aligned and full 4-element group
    int out_offset = batch_idx * (C_out * L_out) + out_ch * L_out + base_out_pos;
    int remaining = ((base_out_pos + 4) <= L_out) ? 4 : (L_out - base_out_pos);
    if (remaining == 4 && ((reinterpret_cast<uintptr_t>(&y[out_offset]) & 15) == 0)) {
        store_float4(&y[out_offset], output);
    } else {
        for (int i = 0; i < remaining; ++i) {
            y[out_offset + i] = acc[i];
        }
    }
}

// Host wrapper function to set up kernel launch parameters
at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& w,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");

    // Get input dimensions
    auto x_sizes = x.sizes();
    int N    = x_sizes[0];
    int C_in = x_sizes[1];
    int L_in = x_sizes[2];

    auto w_sizes = w.sizes();
    int C_out = w_sizes[0];
    int K     = w_sizes[2];

    // Compute output length based on convolution parameters
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Configure block and grid dimensions; each thread processes 4 output positions
    const int threads_per_block = 128;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        C_out, // one grid block per output channel
        (L_out + threads_per_block * 4 - 1) / (threads_per_block * 4), // groups of 4 output positions per thread
        N  // one grid layer per batch element
    );

    conv1d_forward_kernel_aligned_ldg<<<grid_dim, block_dim>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, L_in, C_out, K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_aligned_ldg failed: ", cudaGetErrorString(err));

    return y;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](at::Tensor x,
           at::Tensor w,
           py::object bias_obj,
           int64_t stride,
           int64_t padding,
           int64_t dilation,
           int64_t groups) {
            c10::optional<at::Tensor> bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<at::Tensor>();
            }
            return conv1d_forward_impl(x, w, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CUDA) using __ldg() for optimized read-only loads and 128-bit aligned accesses"
    );
}
