#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp size constant
constexpr int WARP_SIZE = 32;

// Device function to compute partial results within a warp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function to compute a single output element using warp-level optimizations
__device__ float compute_output_element_warp(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n, int c_out, int l_out)
{
    const int lane_id = threadIdx.x % WARP_SIZE;
    float partial_sum = 0.0f;

    // Initialize with bias if present
    if (lane_id == 0) {
        partial_sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    }
    partial_sum = __shfl_sync(0xffffffff, partial_sum, 0);

    // Distribute work across warp lanes
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_w = lane_id; k_w < K_w; k_w += WARP_SIZE) {
            int l_in_nom = l_out + padding - k_w * dilation;
            int l_in = (l_in_nom >= 0) ? l_in_nom / stride : -1;
            
            float contrib = 0.0f;
            if (l_in_nom % stride == 0 && l_in >= 0 && l_in < L_in) {
                float x_val = x[n * C_in * L_in + c_in * L_in + l_in];
                float w_val = weight[c_in * C_out * K_w + c_out * K_w + k_w];
                contrib = x_val * w_val;
            }
            
            // Sum contributions within the warp
            float warp_sum = warp_reduce_sum(contrib);
            if (lane_id == 0) {
                partial_sum += warp_sum;
            }
            partial_sum = __shfl_sync(0xffffffff, partial_sum, 0);
        }
    }

    return partial_sum;
}

__global__ void conv_transpose1d_kernel_warp(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation)
{
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int total_warps = (N * C_out * L_out + WARP_SIZE - 1) / WARP_SIZE;
    
    if (warp_idx >= total_warps) return;

    // Calculate output position
    int flat_idx = warp_idx * WARP_SIZE;
    int l_out = flat_idx % L_out;
    int c_out = (flat_idx / L_out) % C_out;
    int n = flat_idx / (L_out * C_out);

    if (n < N && c_out < C_out && l_out < L_out) {
        float result = compute_output_element_warp(
            x, weight, bias,
            N, C_in, C_out, L_in, L_out, K_w,
            stride, padding, dilation,
            n, c_out, l_out);
            
        // Only the first thread in the warp writes the result
        if (threadIdx.x % WARP_SIZE == 0) {
            y[n * C_out * L_out + c_out * L_out + l_out] = result;
        }
    }
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

    // Launch kernel with warp-based configuration
    int warps_per_block = 8;
    int threads_per_block = warps_per_block * WARP_SIZE;
    int total_elements = N * C_out * L_out;
    int total_warps = (total_elements + WARP_SIZE - 1) / WARP_SIZE;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    conv_transpose1d_kernel_warp<<<blocks, threads_per_block>>>(
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