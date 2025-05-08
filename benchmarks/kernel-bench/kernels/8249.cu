#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Optimized 1D convolution CUDA kernel using __ldg() for read-only accesses
// and assuming 128-bit aligned data for faster global memory loads.
// ---------------------------------------------------------------------
__global__ void conv1d_forward_kernel_opt(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // may be null
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
)
{
    // Each thread computes one output element: (n, out_ch, out_pos).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int out_pos = idx % L_out;
    int out_ch  = (idx / L_out) % C_out;
    int n       = idx / (L_out * C_out);

    // Determine group index based on channel grouping
    int group_size_out = C_out / groups;
    int group_size_in  = C_in  / groups;
    int group_idx      = out_ch / group_size_out;

    float val = 0.0f;
    // Convolution accumulation using __ldg() for read-only accesses
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                int x_index = n * (C_in * L_in) + in_ch * L_in + in_pos;
                int w_index = out_ch * (group_size_in * K) + local_in_ch * K + k;
                float x_val = __ldg(&x[x_index]);
                float w_val = __ldg(&w[w_index]);
                val += x_val * w_val;
            }
        }
    }

    // Add bias if provided using __ldg()
    if (bias_ptr) {
        val += __ldg(&bias_ptr[out_ch]);
    }

    // Write result. Assumed to be 128-bit aligned when possible.
    int y_index = n * (C_out * L_out) + out_ch * L_out + out_pos;
    y[y_index] = val;
}

// ---------------------------------------------------------------------
// Implementation of conv1d forward with optional bias using the optimized kernel
// ---------------------------------------------------------------------
at::Tensor conv1d_forward_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

    // x shape: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // weight shape: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    // Create output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    // Bias pointer (may be null if bias is not provided)
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Launch optimized kernel
    int total_threads = N * C_out * L_out;
    int blockSize = 256;
    int gridSize  = (total_threads + blockSize - 1) / blockSize;

    conv1d_forward_kernel_opt<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups,
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_opt failed: ", cudaGetErrorString(err));

    return y;
}

// ---------------------------------------------------------------------
// Pybind11 binding with optional bias under the same module name
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
        "Optimized 1D Convolution forward (CUDA) with optional bias using __ldg() for aligned global memory access"
    );
}
