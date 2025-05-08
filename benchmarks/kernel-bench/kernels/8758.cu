#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel_atomic(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int l_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (l_out >= L_out || c_out >= C_out) return;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_base = n * C_in * L_in + c_in * L_in;
        int w_base = c_in * C_out * K_w + c_out * K_w;

        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride != 0) continue;
            int l_in = l_in_nom / stride;
            if (l_in >= 0 && l_in < L_in) {
                float x_val = x[x_base + l_in];
                float w_val = weight[w_base + k_w];
                value += x_val * w_val;
            }
        }
    }

    atomicAdd(&y[n * C_out * L_out + c_out * L_out + l_out], value);
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

    auto y = torch::zeros({N, C_out, L_out}, x.options());

    dim3 threads(16, 16);
    dim3 blocks((L_out + threads.x - 1) / threads.x, (C_out + threads.y - 1) / threads.y, N);

    conv_transpose1d_kernel_atomic<<<blocks, threads>>>(
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
        "Optimized Conv Transpose1D with atomic operations",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
