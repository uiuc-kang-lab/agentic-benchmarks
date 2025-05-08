#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for weights and bias
__constant__ float c_weight[16384];  // Adjust size based on expected maximum weight size
__constant__ float c_bias[1024];     // Adjust size based on expected maximum bias size

__global__ void conv_transpose1d_kernel_const_mem(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    bool has_bias) {

    int total_elements = N * C_out * L_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_elements = blockDim.x * gridDim.x;

    for (int idx = index; idx < total_elements; idx += stride_elements) {
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        float value = has_bias ? c_bias[c_out] : 0.0f;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            int w_base = c_in * C_out * K_w + c_out * K_w;

            #pragma unroll 4
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int l_in_nom = l_out + padding - k_w * dilation;
                int l_in = l_in_nom / stride;
                
                if ((l_in_nom % stride == 0) && (l_in >= 0) && (l_in < L_in)) {
                    float x_val = x[x_base + l_in];
                    float w_val = c_weight[w_base + k_w];
                    value += x_val * w_val;
                }
            }
        }
        y[n * C_out * L_out + c_out * L_out + l_out] = value;
    }
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1) {

    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    bool has_bias = !bias_obj.is_none();
    
    // Copy weight to constant memory
    int weight_size = weight.numel() * sizeof(float);
    TORCH_CHECK(weight_size <= 16384 * sizeof(float), "Weight tensor too large for constant memory");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

    // Copy bias to constant memory if present
    if (has_bias) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        cudaMemcpyToSymbol(c_bias, bias.data_ptr<float>(), bias.numel() * sizeof(float));
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    int threads = 256;
    int blocks = (N * C_out * L_out + threads - 1) / threads;

    conv_transpose1d_kernel_const_mem<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation,
        has_bias);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv_transpose1d_forward,
        "Constant Memory Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}