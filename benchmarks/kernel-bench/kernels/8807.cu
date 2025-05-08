#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the warp size constant
constexpr int WARP_SIZE = 32;

// Kernel: Each warp computes one output element using warp-level reduction
__global__ void conv_transpose1d_warp_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Calculate the number of warps per block and lane id
    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Global warp id; each warp is assigned one output element
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    int total_outputs = N * C_out * L_out;
    int total_warps = gridDim.x * warps_per_block;

    // Loop over output elements assigned to this warp in a grid-stride fashion
    for (int idx = global_warp_id; idx < total_outputs; idx += total_warps) {
        // Decode the output indices
        int l_out = idx % L_out;
        int temp = idx / L_out;
        int c_out = temp % C_out;
        int n = temp / C_out;

        // Each warp will reduce over all kernel operations across C_in and K_w
        int total_ops = C_in * K_w;
        float sum = 0.0f;

        // Partition the work among the 32 threads in the warp
        for (int i = lane; i < total_ops; i += WARP_SIZE) {
            int c_in = i / K_w;
            int k_w = i % K_w;
            int l_in_nom = l_out + padding - k_w * dilation;
            // Only accumulate if the computed index aligns with the stride
            if ((l_in_nom % stride) == 0) {
                int l_in = l_in_nom / stride;
                if (l_in >= 0 && l_in < L_in) {
                    float x_val = x[n * (C_in * L_in) + c_in * L_in + l_in];
                    float w_val = weight[c_in * (C_out * K_w) + c_out * K_w + k_w];
                    sum += x_val * w_val;
                }
            }
        }

        // Warp-level reduction using __shfl_down_sync to sum across lanes
        unsigned int mask = 0xffffffff;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        // The first lane writes back the result after adding bias if available
        if (lane == 0) {
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            y[n * (C_out * L_out) + c_out * L_out + l_out] = sum;
        }
    }
}

// Host function to prepare tensors and launch the kernel
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

    // Compute the output length
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;
    auto y = torch::empty({N, C_out, L_out}, x.options());

    int total_outputs = N * C_out * L_out;
    // Launch configuration: use a multiple of WARP_SIZE for optimal warp occupancy (e.g., 256 threads per block)
    int threads = 256;
    int warps_per_block = threads / WARP_SIZE;
    int blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

    conv_transpose1d_warp_kernel<<<blocks, threads>>>(
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
        "Conv Transpose1D forward (CUDA) with warp-level reduction using __shfl_down_sync",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
