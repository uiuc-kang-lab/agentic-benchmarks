#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>

namespace py = pybind11;

// -----------------------------------------------------------------
// 1D convolution CUDA kernel with efficient thread and block mapping
// -----------------------------------------------------------------
__global__ void conv1d_forward_kernel_efficient(
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
) {
    // Grid configuration: each block handles one (n, out_ch) pair
    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int out_ch = blockIdx.y;
    int n = blockIdx.z;

    if (out_pos >= L_out) return;

    int group_size_out = C_out / groups;
    int group_size_in  = C_in / groups;
    int group_idx = out_ch / group_size_out;

    float sum = 0.0f;
    #pragma unroll
    for (int local_in_ch = 0; local_in_ch < group_size_in; local_in_ch++) {
        int in_ch = group_idx * group_size_in + local_in_ch;
        for (int k = 0; k < K; k++) {
            int in_pos = out_pos * stride + k * dilation - padding;
            // Clamp index
            int clamped_in_pos = in_pos < 0 ? 0 : (in_pos >= L_in ? (L_in - 1) : in_pos);
            // Mask to zero out values that are actually out of bounds
            float mask = ((unsigned)in_pos < (unsigned)L_in) ? 1.0f : 0.0f;
            int x_index = n * (C_in * L_in) + in_ch * L_in + clamped_in_pos;
            int w_index = out_ch * (group_size_in * K) + local_in_ch * K + k;
            sum += mask * x[x_index] * w[w_index];
        }
    }
    if (bias_ptr) {
        sum += bias_ptr[out_ch];
    }
    int y_index = n * (C_out * L_out) + out_ch * L_out + out_pos;
    y[y_index] = sum;
}

// -------------------------------------------------------
// Implementation of conv1d forward with efficient thread and block mapping
// -------------------------------------------------------
at::Tensor conv1d_forward_impl_efficient(
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

    // Input dimensions: [N, C_in, L_in]
    auto x_sizes = x.sizes();
    int64_t N    = x_sizes[0];
    int64_t C_in = x_sizes[1];
    int64_t L_in = x_sizes[2];

    // Weight dimensions: [C_out, C_in/groups, K]
    auto w_sizes = weight.sizes();
    int64_t C_out = w_sizes[0];
    int64_t K     = w_sizes[2];

    // Calculate output length
    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    TORCH_CHECK(L_out > 0, "Calculated output length is non-positive.");

    auto y = torch::empty({N, C_out, L_out}, x.options().dtype(at::kFloat));

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "bias must be a CUDA tensor if provided");
        TORCH_CHECK(bias_opt.value().scalar_type() == at::kFloat, "bias must be float32");
        bias_ptr = bias_opt.value().data_ptr<float>();
    }

    // Launch configuration: use 3D grid and 1D blocks
    dim3 threads(256);
    dim3 blocks((L_out + threads.x - 1) / threads.x, C_out, N);

    conv1d_forward_kernel_efficient<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        (int)N, (int)C_in, (int)L_in, (int)C_out, (int)K,
        (int)stride, (int)padding, (int)dilation, (int)groups, (int)L_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1d_forward_kernel_efficient failed: ", cudaGetErrorString(err));

    return y;
}

// -----------------------------------------------------
// Pybind11 binding for the efficient convolution kernel
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
            return conv1d_forward_impl_efficient(x, weight, bias, stride, padding, dilation, groups);
        },
        "1D Convolution forward (CUDA) with efficient thread and block mapping"
    );
}