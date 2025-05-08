#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses __ldg() for read-only accesses to promote 128-bit aligned loads
// from global memory via the read-only cache. It assumes that the input tensors are allocated
// with proper alignment (typically 16-byte aligned) by PyTorch.

__global__ void conv_transpose1d_kernel_shared(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * C_out * L_out) return;

    // Compute indices for output y
    int n = tid / (C_out * L_out);
    int rem = tid % (C_out * L_out);
    int c_out = rem / L_out;
    int l_out = rem % L_out;

    // Use __ldg() for bias load if available
    float value = (bias != nullptr) ? __ldg(&bias[c_out]) : 0.0f;

    int x_base = n * C_in * L_in;
    int w_base = c_out * K_w;

    // Loop over input channels and kernel width
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_offset = x_base + c_in * L_in;
        int w_offset = c_in * C_out * K_w + w_base;
        for (int k_w = 0; k_w < K_w; ++k_w) {
            int l_in_nom = l_out + padding - k_w * dilation;
            if (l_in_nom % stride == 0) {
                int l_in = l_in_nom / stride;
                if (l_in >= 0 && l_in < L_in) {
                    // Use __ldg() for read-only x and weight accesses
                    float x_val = __ldg(&x[x_offset + l_in]);
                    float w_val = __ldg(&weight[w_offset + k_w]);
                    value += x_val * w_val;
                }
            }
        }
    }
    
    // Write the result to y; writes are coalesced as each thread writes one element
    y[tid] = value;
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
        auto bias = bias_obj.cast<torch::Tensor>().contiguous();
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

    int total_elements = N * C_out * L_out;
    int threads = 256;  // threads per block
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel_shared<<<blocks, threads>>>(
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
        "Conv Transpose1D forward (CUDA) with __ldg() and aligned loads",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
