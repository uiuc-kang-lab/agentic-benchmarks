#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the output value for a single output element
__device__ inline float compute_output_value(
    const float* __restrict__ x,       // Input tensor: [N, C_in, L_in]
    const float* __restrict__ weight,  // Weight tensor: [C_in, C_out, K_w]
    const float* __restrict__ bias,    // Bias tensor: [C_out] or nullptr
    int n, int c_out, int l_out,         // Output indices
    int C_in, int L_in, int K_w,
    int stride, int padding, int dilation) {

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over input channels and kernel positions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // Compute base offsets
        int x_offset = n * (C_in * L_in) + c_in * L_in;
        int w_offset = c_in * (/*C_out*/  /* note: using parameter c_out base offset */  + 0) ;
        // Actually, use the formula: w_index = c_in * (C_out*K_w) + c_out*K_w + k_w
        int weight_base = c_in * (/*C_out*K_w*/ C_out * K_w) + c_out * K_w;
        
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride != 0)
                continue;
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                value += x[x_offset + l_in] * weight[weight_base + k_w];
            }
        }
    }
    return value;
}

// Modular kernel that uses the device function to compute each output element
__global__ void conv_transpose1d_kernel_modular(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * L_out;

    // Grid-stride loop over all output elements
    for (int index = idx; index < total_elements; index += blockDim.x * gridDim.x) {
        int l_out = index % L_out;
        int c_out = (index / L_out) % C_out;
        int n = index / (L_out * C_out);

        y[n * C_out * L_out + c_out * L_out + l_out] =
            compute_output_value(x, weight, bias,
                                 n, c_out, l_out,
                                 C_in, L_in, K_w,
                                 stride, padding, dilation);
    }
}

// Host function to set up and launch the CUDA kernel
torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    // Convert inputs to torch::Tensor and ensure they are contiguous
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

    // Get tensor dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Launch kernel with grid-stride loop
    int total_elements = N * C_out * L_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel_modular<<<blocks, threads>>>(
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
        "Modular Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
