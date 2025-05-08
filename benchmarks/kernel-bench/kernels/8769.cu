#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using __ldg() for read-only accesses and aligning memory accesses to 128-bit boundaries
__global__ void conv_transpose1d_kernel_optimized_ldg(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int total_elements = N * C_out * L_out;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_elements = blockDim.x * gridDim.x;

    // Loop over each output element using grid-stride loop
    for (int idx = tid; idx < total_elements; idx += stride_elements) {
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        // Load bias using __ldg if available
        float result = (bias != nullptr) ? __ldg(&bias[c_out]) : 0.0f;

        // Loop over the input channels
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            int w_base = c_in * C_out * K_w + c_out * K_w;

            // Process kernel positions in groups of 4 for vectorized load if possible
            int k4 = (K_w / 4) * 4;  // largest multiple of 4 less than or equal to K_w
            for (int k = 0; k < k4; k += 4) {
                // Load 4 weight elements as a float4 using __ldg for read-only data
                float4 w_vec = __ldg(reinterpret_cast<const float4*>(&weight[w_base + k]));
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int curr_k = k + i;
                    int l_in_nom = l_out + padding - curr_k * dilation;
                    // Check if l_in_nom is valid: divisible by stride and in range
                    if ((l_in_nom % stride) == 0) {
                        int l_in = l_in_nom / stride;
                        if (l_in >= 0 && l_in < L_in) {
                            float x_val = __ldg(&x[x_base + l_in]);
                            // Access the i-th element from the float4 vector
                            float w_val = ((float*)&w_vec)[i];
                            result += x_val * w_val;
                        }
                    }
                }
            }
            // Handle remaining kernel positions that are not a multiple of 4
            for (int k = k4; k < K_w; ++k) {
                int l_in_nom = l_out + padding - k * dilation;
                if ((l_in_nom % stride) == 0) {
                    int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        float x_val = __ldg(&x[x_base + l_in]);
                        float w_val = __ldg(&weight[w_base + k]);
                        result += x_val * w_val;
                    }
                }
            }
        }
        // Write the computed result to output
        y[n * C_out * L_out + c_out * L_out + l_out] = result;
    }
}


// Pybind11 binding for Python
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

    conv_transpose1d_kernel_optimized_ldg<<<blocks, threads>>>(
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
        "Optimized Conv Transpose1D forward (CUDA) using __ldg() and 128-bit aligned loads",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
