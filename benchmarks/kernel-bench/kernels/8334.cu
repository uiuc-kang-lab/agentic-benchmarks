#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// -----------------------------------------------------
// Optimized 1D convolution CUDA kernel with memory coalescing
// -----------------------------------------------------
__global__ void conv1d_forward_kernel_coalesced(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias_ptr, // can be null if no bias
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
    // Calculate thread index within grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    // Shared memory for input and weights
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_w = &shared_mem[C_in * L_in];

    // Load input and weights into shared memory
    for (int i = threadIdx.x; i < C_in * L_in; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    for (int i = threadIdx.x; i < C_out * C_in * K; i += blockDim.x) {
        shared_w[i] = w[i];
    }
    __syncthreads();

    // Calculate output position using coalesced memory access pattern
    int n = idx / (C_out * L_out);  // batch index
    int out_ch = (idx / L_out) % C_out;  // output channel
    int out_pos = idx % L_out;  // position in output vector

    // Determine group index for this output channel
    int group_size_out = C_out / groups;  
    int group_size_in  = C_in  / groups;  
    int group_idx      = out_ch / group_size_out;

    float val = 0.0f;
    // Convolution accumulation using coalesced memory access
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            if (in_pos >= 0 && in_pos < L_in) {
                float x_val = x[n * (C_in * L_in) + in_ch * L_in + in_pos];
                float w_val = w[out_ch * (group_size_in * K)
                                + local_in_ch * K
                                + k];
                val += x_val * w_val;
            }
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        val += bias_ptr[out_ch];
    }

    // Store result ensuring coalesced write pattern
    y[idx] = val;
}

// -------------------------------------------------------
// Implementation of conv1d forward with optional bias
// -------------------------------------------------------
at::Tensor conv1d_forward_impl_coalesced(
    const at::Tensor& x,
    const at::Tensor& weight,
    c10::optional<at::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
)
{
    // Check device, dtype
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

    // Calculate conv1d output length
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

    // Launch kernel
    int total_threads = N * C_out * L_out;
    int blockSize = 256;
    int gridSize  = (total_threads + blockSize - 1) / blockSize;

    conv1d_forward_kernel_coalesced<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, 
        (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_coalesced failed: ", cudaGetErrorString(err));

    return y;
}

// -----------------------------------------------------
// Pybind11 binding with optional bias
// -----------------------------------------------------
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
            return conv1d_forward_impl_coalesced(x, weight, bias, stride, padding, dilation, groups);
        },
        "Optimized 1D Convolution forward (CUDA) with memory coalescing"
    );
}
