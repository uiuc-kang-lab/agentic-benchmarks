#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Helper function to load a float from global memory with __ldg() and
// attempt to use vectorized (128-bit) loads if the address is aligned
// ---------------------------------------------------------------------
__device__ inline float load_aligned_val(const float* ptr, int idx) {
    // Check if the address (ptr + idx) is 16-byte (128-bit) aligned
    if ((((uintptr_t)(ptr + idx)) & 0xF) == 0) {
        int vec_idx = idx >> 2; // equivalent to idx / 4
        int rem = idx & 3;      // equivalent to idx % 4
        const float4* vec_ptr = reinterpret_cast<const float4*>(ptr);
        float4 vec = __ldg(vec_ptr + vec_idx);
        // Access the appropriate component
        return ((float*)&vec)[rem];
    } else {
        // Fallback to scalar load
        return __ldg(ptr + idx);
    }
}

// ---------------------------------------------------------------------
// Optimized 1D convolution CUDA kernel that uses __ldg() and aligned loads
// for read-only global memory accesses. Each thread computes one output
// element for the standard 1D convolution operation.
// ---------------------------------------------------------------------
__global__ void conv1d_forward_kernel_aligned(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be null if no bias
    float* __restrict__ y,
    const int N,         // batch size
    const int C_in,      // input channels
    const int L_in,      // input length
    const int C_out,     // output channels
    const int K,         // kernel size
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int L_out      // output length
) {
    // Each thread computes one output element indexed by (n, out_ch, out_pos)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int out_pos = idx % L_out;
    int out_ch  = (idx / L_out) % C_out;
    int n       = idx / (L_out * C_out);

    // Determine input channel grouping
    int group_size_out = C_out / groups;
    int group_size_in  = C_in  / groups;
    int group_idx      = out_ch / group_size_out;

    float val = 0.0f;
    // Loop over the convolution kernel: each output element is computed by
    // summing over a subset of input channels and kernel weights
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                int x_index = n * (C_in * L_in) + in_ch * L_in + in_pos;
                int w_index = out_ch * (group_size_in * K) + local_in_ch * K + k;
                float x_val = load_aligned_val(x, x_index);
                float w_val = load_aligned_val(w, w_index);
                val += x_val * w_val;
            }
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        val += load_aligned_val(bias_ptr, out_ch);
    }

    int y_index = n * (C_out * L_out) + out_ch * L_out + out_pos;
    y[y_index] = val;
}

// ---------------------------------------------------------------------
// Host implementation: sets up tensor dimensions and launches the kernel
// ---------------------------------------------------------------------
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

    // x: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // weight: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate the output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Create the output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    // Handle optional bias
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Launch configuration
    int total_threads = N * C_out * L_out;
    int blockSize = 256;
    int gridSize  = (total_threads + blockSize - 1) / blockSize;

    conv1d_forward_kernel_aligned<<<gridSize, blockSize>>>(
         x.data_ptr<float>(),
         weight.data_ptr<float>(),
         bias_ptr,
         y.data_ptr<float>(),
         (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
         (int)stride, (int)padding, (int)dilation, (int)groups,
         (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_aligned failed: ", cudaGetErrorString(err));

    return y;
}

// ---------------------------------------------------------------------
// Pybind11 binding: exposes the convolution forward function with optional bias
// ---------------------------------------------------------------------
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
       "Optimized 1D Convolution forward (CUDA) with __ldg() and 128-bit aligned loads"
    );
}
