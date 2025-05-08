#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for ConvTranspose1D using shared memory for weights
__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Allocate shared memory for the weight tensor
    extern __shared__ float s_weight[];

    int total_weight = C_in * C_out * K_w;
    int tid = threadIdx.x;
    // Cooperative loading of the weight tensor into shared memory
    for (int i = tid; i < total_weight; i += blockDim.x) {
        s_weight[i] = weight[i];
    }

    // Synchronize threads to ensure the weight tensor is fully loaded
    __syncthreads();

    // Compute global output index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * L_out;
    if (index >= total_elements) return;

    // Calculate indices (n, c_out, l_out) from the flattened index
    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);

    // Initialize output with bias if provided
    float value = 0.0f;
    if (bias != nullptr) {
        value = bias[c_out];
    }

    // Perform the transposed convolution using the shared memory weights
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride != 0) continue;
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
                // Access the weight from shared memory; layout: [C_in, C_out, K_w]
                float w_val = s_weight[c_in * (C_out * K_w) + c_out * K_w + k_w];
                value += x_val * w_val;
            }
        }
    }
    y[n * C_out * L_out + c_out * L_out + l_out] = value;
}

// Host function for launching the optimized kernel
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,            // x: torch::Tensor
    py::object weight_obj,       // weight: torch::Tensor or torch::nn::Parameter
    py::object bias_obj = py::none(),  // bias: torch::Tensor or None
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    // Convert py::object to torch::Tensor and ensure they are contiguous and CUDA-based
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Get dimensions from the input tensors
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Calculate dynamic shared memory size (in bytes) for weight caching
    int sharedMemBytes = C_in * C_out * K_w * sizeof(float);

    // Launch parameters
    int total_elements = N * C_out * L_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel_optimized<<<blocks, threads, sharedMemBytes>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);

    // Check for CUDA errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Optimized Conv Transpose1D forward (CUDA) using shared memory for weights",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
