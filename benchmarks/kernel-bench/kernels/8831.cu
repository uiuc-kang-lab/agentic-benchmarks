#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// Converts a linear thread index to (n, c_out, l_out) coordinates
__device__ __forceinline__ void linear_to_3d_index(int idx, int L_out, int C_out, int &n, int &c_out, int &l_out) {
    l_out = idx % L_out;
    c_out = (idx / L_out) % C_out;
    n = idx / (L_out * C_out);
}

// Computes the input index for a given output location and kernel offset
// Returns true if the computed input index is valid, and outputs it in l_in
__device__ __forceinline__ bool calc_input_index(int l_out, int k_w, int stride, int dilation, int padding, int L_in, int &l_in) {
    int l_in_nom = l_out + padding - k_w * dilation;
    if (l_in_nom % stride != 0) {
        return false;
    }
    l_in = l_in_nom / stride;
    return (l_in >= 0 && l_in < L_in);
}

// Modular device function to compute the convolution result for one output element
__device__ __forceinline__ float compute_output_element(
    const float* __restrict__ x,     // Input tensor [N, C_in, L_in]
    const float* __restrict__ weight, // Weight tensor [C_in, C_out, K_w]
    const float* __restrict__ bias,   // Bias tensor [C_out] or nullptr
    int n, int C_in, int C_out, int l_out,
    int L_in, int K_w, int c_out,
    int stride, int padding, int dilation) {

    float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;
    int l_in = 0;
    
    // Loop over each input channel
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Calculate base indices for input and weight
        int x_base = n * C_in * L_in + c_in * L_in;
        int w_base = c_in * C_out * K_w + c_out * K_w;
        
        // Loop over kernel width positions
        for (int k = 0; k < K_w; ++k) {
            if (calc_input_index(l_out, k, stride, dilation, padding, L_in, l_in)) {
                float x_val = x[x_base + l_in];
                float w_val = weight[w_base + k];
                out_val += x_val * w_val;
            }
        }
    }
    return out_val;
}

// CUDA kernel for 1D transposed convolution
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out,
    int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int n, c_out, l_out;
    linear_to_3d_index(idx, L_out, C_out, n, c_out, l_out);
    
    y[idx] = compute_output_element(x, weight, bias, n, C_in, C_out, l_out, L_in, K_w, c_out, stride, padding, dilation);
}

// Frontend function exposed to Python via PyBind11
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    // Convert input objects to contiguous tensors
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    // Tensor dimensions from input and weight
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;
    auto y = torch::empty({N, C_out, L_out}, x.options());

    int total_elements = N * C_out * L_out;
    int threads = 256;
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
    m.def("forward",
          &conv_transpose1d_forward,
          "ConvTranspose1D forward (CUDA, modular device functions)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1);
}
