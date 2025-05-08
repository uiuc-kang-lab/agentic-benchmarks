#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel_aligned(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int total_elements = N * C_out * L_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_elements = blockDim.x * gridDim.x;

    // Ensure 128-bit alignment for memory accesses
    static_assert(sizeof(float4) == 16, "float4 should be 16 bytes");

    for (int idx = index; idx < total_elements; idx += stride_elements) {
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        float value = (bias != nullptr) ? __ldg(&bias[c_out]) : 0.0f;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            int w_base = c_in * C_out * K_w + c_out * K_w;

            // Process kernel positions in groups of 4 when possible
            int k_w_aligned = K_w & ~3;  // Round down to multiple of 4
            
            // Handle aligned kernel positions
            for (int k_w = 0; k_w < k_w_aligned; k_w += 4) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int curr_k = k_w + k;
                    int l_in_nom = l_out + padding - curr_k * dilation;
                    int mod_cond = (l_in_nom % stride == 0);
                    int l_in = l_in_nom / stride;
                    int range_cond = (l_in >= 0 && l_in < L_in);
                    int valid = mod_cond * range_cond;

                    float x_val = valid ? __ldg(&x[x_base + l_in]) : 0.0f;
                    float w_val = __ldg(&weight[w_base + curr_k]);
                    value += x_val * w_val;
                }
            }

            // Handle remaining kernel positions
            for (int k_w = k_w_aligned; k_w < K_w; ++k_w) {
                int l_in_nom = l_out + padding - k_w * dilation;
                int mod_cond = (l_in_nom % stride == 0);
                int l_in = l_in_nom / stride;
                int range_cond = (l_in >= 0 && l_in < L_in);
                int valid = mod_cond * range_cond;

                float x_val = valid ? __ldg(&x[x_base + l_in]) : 0.0f;
                float w_val = __ldg(&weight[w_base + k_w]);
                value += x_val * w_val;
            }
        }

        // Store result
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

    // Optimize thread block size for aligned memory access
    int threads = 256;
    int blocks = (N * C_out * L_out + threads - 1) / threads;

    conv_transpose1d_kernel_aligned<<<blocks, threads>>>(
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
        "Aligned memory Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}