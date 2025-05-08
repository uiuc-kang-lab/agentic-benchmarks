#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Force inline device function with FMA optimization
__device__ __forceinline__ float compute_output_element(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n, int c_out, int l_out)
{
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride == 0) {
                int l_in = l_in_nom / stride;
                if (l_in >= 0 && l_in < L_in) {
                    const float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
                    const float w_val = weight[c_in * C_out * K_w + c_out * K_w + k_w];
                    value = __fmaf_rn(x_val, w_val, value);
                }
            }
        }
    }
    return value;
}

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_out * L_out) return;

    const int l_out = index % L_out;
    const int c_out = (index / L_out) % C_out;
    const int n = index / (L_out * C_out);

    y[index] = compute_output_element(x, weight, bias, N, C_in, C_out, L_in, L_out, K_w,
                                     stride, padding, dilation, n, c_out, l_out);
}

torch::Tensor conv_transpose1d_forward(
    py::object x_obj,
    py::object weight_obj,
    py::object bias_obj = py::none(),
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1)
{
    torch::Tensor x = x_obj.cast<torch::Tensor>().contiguous();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>().contiguous();
    float* bias_ptr = nullptr;

    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int L_in = x.size(2);
    const int K_w = weight.size(2);
    const int C_out = weight.size(1);
    const int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const int threads = 512;
    const int elements = N * C_out * L_out;
    const int blocks = (elements + threads - 1) / threads;

    conv_transpose1d_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                              weight.data_ptr<float>(),
                                              bias_ptr,
                                              y.data_ptr<float>(),
                                              N, C_in, C_out, L_in, L_out, K_w,
                                              stride, padding, dilation);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose1d_forward, "Conv Transpose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride") = 1, py::arg("padding") = 0, py::arg("dilation") = 1);
}