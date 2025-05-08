#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ConvTranspose1D with identical computation
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_ = blockDim.x * gridDim.x;
    int total_elements = N * C_out * L_out;
    for (int index = i; index < total_elements; index += stride_) {

    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);

    float value = 0.0f;
    if (bias != nullptr) {
        value = bias[c_out];
    }

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride != 0) continue;
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
                float w_val = weight[c_in * C_out * K_w + c_out * K_w + k_w];
                value += x_val * w_val;
            }
        }
    }

    y[n * C_out * L_out + c_out * L_out + l_out] = value;
}

// Forward function with a tunable block size parameter (default 256)
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // Input tensor x
    py::object weight_obj,       // Weight tensor
    py::object bias_obj = py::none(),  // Bias tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int block_size = 256) {      // Tunable block size for experimentation

    // Convert py::object to torch::Tensor and ensure contiguous CUDA tensors
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Get dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Determine kernel launch configuration using the tunable block size
    int total_elements = N * C_out * L_out;
    int threads = block_size;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Tuned Conv Transpose1D forward (CUDA) with configurable block size for optimization",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1,
        py::arg("block_size") = 256);
}
