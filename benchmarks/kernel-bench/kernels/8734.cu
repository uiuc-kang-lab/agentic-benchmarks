#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    const int total_elements = N * C_out * L_out;

    // Stride loop to handle multiple elements per thread
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int l_out = idx % L_out;
        const int c_out = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);

        float value = bias != nullptr ? bias[c_out] : 0.0f;

        #pragma unroll 4
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int x_batch_offset = n * C_in * L_in + c_in * L_in;
            const int w_channel_offset = c_in * C_out * K_w + c_out * K_w;

            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int l_in_nom = l_out + padding - k_w * dilation;
                if (l_in_nom % stride == 0) {
                    const int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        value += x[x_batch_offset + l_in] * 
                                weight[w_channel_offset + k_w];
                    }
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

    torch::Tensor x = x_obj.cast<torch::Tensor>();
    torch::Tensor weight = weight_obj.cast<torch::Tensor>();

    x = x.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");

    float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>();
        bias = bias.contiguous();
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

    const int threads = 256;
    const int max_blocks = 65535;
    const int min_elements_per_thread = 4;
    const int total_elements = N * C_out * L_out;
    
    int blocks = std::min(max_blocks, 
                         (total_elements + threads * min_elements_per_thread - 1) / 
                         (threads * min_elements_per_thread));

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
        "Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}