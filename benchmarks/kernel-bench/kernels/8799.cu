#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size constant
constexpr int WARP_SIZE = 32;

// Optimized kernel: Combines warp-level work assignment with grid-stride partitioning for balanced workload
__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Each thread block is organized in warps
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Total number of warps in the grid
    int total_warps = gridDim.x * warps_per_block;
    int total_elements = N * C_out * L_out;
    // Evenly divide the work among all warps
    int elems_per_warp = (total_elements + total_warps - 1) / total_warps;
    int warp_start = global_warp_id * elems_per_warp;
    int warp_end = (warp_start + elems_per_warp < total_elements) ? (warp_start + elems_per_warp) : total_elements;

    // Each warp's lanes iterate over contiguous segments with stride = WARP_SIZE
    #pragma unroll
    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        // Compute output indices
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n = idx / (L_out * C_out);

        // Initialize output with bias if provided
        float out_val = (bias != nullptr) ? bias[c_out] : 0.0f;

        int x_batch_offset = n * C_in * L_in;
        // Pre-compute base offset for input spatial index
        int l_in_nom_base = l_out + padding;
        // Pre-compute weight offset for current output channel
        int w_out_offset = c_out * K_w;

        // Loop over input channels and kernel width
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_channel_offset = x_batch_offset + c_in * L_in;
            int w_offset = c_in * C_out * K_w + w_out_offset;
            #pragma unroll
            for (int k = 0; k < K_w; ++k) {
                int l_in_nom = l_in_nom_base - k * dilation;
                // Check if the position aligns with the stride and lies within input bounds
                if ((l_in_nom % stride) == 0) {
                    int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        out_val += x[x_channel_offset + l_in] * weight[w_offset + k];
                    }
                }
            }
        }
        y[idx] = out_val;
    }
}

// Host function: validates tensors, computes output dimensions, and launches the kernel
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
    torch::Tensor bias;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
        bias = bias.contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int K_w = weight.size(2);
    int C_out = weight.size(1);
    // Compute output length based on conv-transpose formula
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    // Configure launch parameters with threads as a multiple of warp size
    int threads_per_block = 256;  // 256 is divisible by 32
    int total_elements = N * C_out * L_out;
    // We launch enough blocks so that each warp gets a balanced portion
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose1d_kernel_optimized<<<blocks, threads_per_block>>>(
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
        "Optimized Conv Transpose1D forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1);
}
