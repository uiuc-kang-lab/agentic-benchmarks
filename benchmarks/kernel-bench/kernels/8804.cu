#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp size constant
constexpr int WARP_SIZE = 32;

__global__ void optimized_conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    int total_elements = N * C_out * L_out;
    int elements_per_thread = (total_elements + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    
    for (int idx = global_warp_id * elements_per_thread + lane_id; idx < total_elements; idx += WARP_SIZE * gridDim.x * blockDim.x) {
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        float value = bias ? bias[c_out] : 0.0f;

        int l_in_nom_start = l_out + padding;
        int x_batch_offset = n * C_in * L_in;

        #pragma unroll
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_offset = x_batch_offset + c_in * L_in;
            int w_offset = c_in * C_out * K_w + c_out * K_w;

            for (int k_w = 0; k_w < K_w; ++k_w) {
                // For transposed conv, we need to check if the output position corresponds to a valid input position
                int l_in = (l_out + padding - k_w * dilation);
                if (l_in % stride == 0) {
                    l_in /= stride;
                    if (l_in >= 0 && l_in < L_in) {
                        value += x[x_offset + l_in] * weight[w_offset + (K_w - 1 - k_w)];
                    }
                }
            }
        }

        y[idx] = value;
    }
}

torch::Tensor optimized_conv_transpose1d_forward(
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

    int threads_per_block = 256;
    int blocks = (N * C_out * L_out + threads_per_block - 1) / threads_per_block;

    optimized_conv_transpose1d_kernel<<<blocks, threads_per_block>>>(
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
        &optimized_conv_transpose1d_forward,
        "Optimized Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}