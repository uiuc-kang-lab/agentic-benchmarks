#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int compute_l_in(int l_out, int k, int stride, int padding, int dilation, int L_in) {
    int l_in_nom = l_out + padding - k * dilation;
    if (l_in_nom % stride != 0)
        return -1;
    int l_in = l_in_nom / stride;
    return (l_in >= 0 && l_in < L_in) ? l_in : -1;
}

__device__ inline float process_channel(const float* x_channel, const float* weight_channel,
                                        int l_out, int L_in, int K_w,
                                        int stride, int padding, int dilation) {
    float sum = 0.0f;
    for (int k = 0; k < K_w; k++) {
        int l_in = compute_l_in(l_out, k, stride, padding, dilation, L_in);
        if (l_in != -1) {
            sum += x_channel[l_in] * weight_channel[k];
        }
    }
    return sum;
}

__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,    // [C_in, C_out, K_w]
    const float* __restrict__ bias,      // [C_out] or nullptr
    float* __restrict__ y,               // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_elements = blockDim.x * gridDim.x;  // Overall stride for grid-stride loop

    for (int idx = index; idx < N * C_out * L_out; idx += stride_elements) {
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        float result = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int c_in = 0; c_in < C_in; c_in++) {
            const float* x_channel = x + n * C_in * L_in + c_in * L_in;
            const float* weight_channel = weight + c_in * (C_out * K_w) + c_out * K_w;
            result += process_channel(x_channel, weight_channel, l_out, L_in, K_w, stride, padding, dilation);
        }

        y[n * C_out * L_out + c_out * L_out + l_out] = result;
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

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
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

    conv_transpose1d_kernel_optimized<<<blocks, threads>>>(
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
        "Optimized Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}