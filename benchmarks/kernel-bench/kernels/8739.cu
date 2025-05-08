#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that minimizes warp divergence by replacing conditional branches
// with arithmetic masks for uniform control flow
__global__ void conv_transpose1d_kernel_no_divergence(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * L_out;
    if (index >= total_elements) return;

    // Derive output indices from flat index
    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);

    // Initialize output using bias if available
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over input channels and kernel positions
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_base = n * C_in * L_in + c_in * L_in;
        int w_base = c_in * C_out * K_w + c_out * K_w;
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            // Replace divergent "if" checks with arithmetic masks
            int mod_cond = ((l_in_nom % stride) == 0) ? 1 : 0;
            int l_in = l_in_nom / stride;
            int range_cond = (l_in >= 0 && l_in < L_in) ? 1 : 0;
            int valid = mod_cond * range_cond;
            // If not valid, valid==0 ensures contribution is zero
            // Use (valid * l_in) so that if valid==0, index used is x_base + 0 (always safe)
            int index_x = x_base + (valid * l_in);
            float x_val = x[index_x] * valid;
            float w_val = weight[w_base + k_w];
            value += x_val * w_val;
        }
    }
    y[n * C_out * L_out + c_out * L_out + l_out] = value;
}

// Host function that sets up and launches the kernel

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    // Convert input objects to contiguous torch::Tensor objects
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
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    // Allocate the output tensor
    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Determine kernel launch configuration
    int total_elements = N * C_out * L_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel_no_divergence<<<blocks, threads>>>(
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
        "Conv Transpose1D forward (CUDA) with minimized warp divergence",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
