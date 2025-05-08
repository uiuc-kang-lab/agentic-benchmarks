#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Branchless clamp using CUDA intrinsics
__device__ __forceinline__ int clamp_int(int val, int lower, int upper) {
    // Using built-in min and max
    return max(lower, min(val, upper));
}

// Device function that computes a single output element in a branchless manner
// It avoids divergent branches by computing a validity mask and using a safe clamped index for memory load.
__device__ float compute_output_element_branchless(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int C_in, int L_in, int K_w,
    int stride, int padding, int dilation,
    int c_out, int l_out, int n, int C_out) {

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Iterate over input channels and kernel width
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_batch_offset = c_in * L_in;  // offset within the batch for this channel
        int w_offset = c_in * C_out * K_w + c_out * K_w;  // offset for the weight corresponding to (c_in, c_out)
        #pragma unroll
        for (int k_w = 0; k_w < K_w; ++k_w) {
            // Compute the nominal input position
            int r = l_out + padding - k_w * dilation;
            // Compute l_in and the remainder in a branchless manner
            int l_in = r / stride;
            int rem = r - l_in * stride;  // equivalent to r % stride
            
            // Instead of branching, compute a validity mask:
            // valid is 1 if r is divisible by stride and l_in is in [0,L_in), else 0.
            int divisible = (rem == 0);
            int in_range = ((l_in >= 0) && (l_in < L_in));
            int valid = divisible * in_range;  // 1 if both conditions are true; 0 otherwise
            
            // Compute a safe index by clamping l_in to [0, L_in-1].
            int l_in_safe = clamp_int(l_in, 0, L_in - 1);
            
            // Only the contribution from valid positions adds to the result;
            // if not valid, the multiplication by 0 ensures no contribution even though we perform a safe memory load.
            float x_val = x[n * C_in * L_in + x_batch_offset + l_in_safe];
            float w_val = weight[w_offset + k_w];
            value += valid * (x_val * w_val);
        }
    }

    return value;
}

// CUDA kernel for ConvTranspose1D using branchless inner loops
__global__ void conv_transpose1d_kernel_branchless(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    // Calculate indices from linear index
    int l_out = idx % L_out;
    int c_out = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    y[idx] = compute_output_element_branchless(
        x, weight, bias,
        C_in, L_in, K_w,
        stride, padding, dilation,
        c_out, l_out, n, C_out);
}

// Frontend function exposed to Python
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

    // Compute output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;
    auto y = torch::empty({N, C_out, L_out}, x.options());

    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose1d_kernel_branchless<<<blocks, threads>>>(
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
        "Conv Transpose1D forward (CUDA, branchless inner loop)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
