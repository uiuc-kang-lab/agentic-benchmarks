#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define warp size
constexpr int WARP_SIZE = 32;

// Kernel: Each block computes multiple output elements using shared memory reduction
__global__ void optimized_conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Align work to warp boundaries
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    
    // Calculate the number of elements each warp processes
    const int total_warps = gridDim.x * warps_per_block;
    const int elements_per_warp = (N * C_out * L_out + total_warps - 1) / total_warps;
    
    // Calculate start and end positions for this warp
    const int warp_start = global_warp_id * elements_per_warp;
    const int warp_end = min(warp_start + elements_per_warp, N * C_out * L_out);

    __shared__ float shared_bias[32]; // Shared memory for bias
    if (bias && lane_id == 0) {
        shared_bias[warp_id] = bias[warp_id];
    }
    __syncthreads();

    // Process elements within the warp
    #pragma unroll 1
    for (int idx = warp_start + lane_id; idx < warp_end; idx += WARP_SIZE) {
        // Calculate position indices
        const int l_out = idx % L_out;
        const int c_out = (idx / L_out) % C_out;
        const int n = idx / (L_out * C_out);
        
        // Initialize output value
        float value = shared_bias[warp_id];
        
        // Pre-calculate constant offsets
        const int x_batch_offset = n * C_in * L_in;
        const int w_cout_offset = c_out * K_w;
        
        // Process input channels
        #pragma unroll 4
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int x_offset = x_batch_offset + c_in * L_in;
            const int w_offset = c_in * C_out * K_w + w_cout_offset;
            
            // Process kernel width
            #pragma unroll
            for (int k_w = 0; k_w < K_w; ++k_w) {
                // Calculate input position
                const int l_in_nom = l_out + padding - k_w * dilation;
                const int l_in = l_in_nom / stride;
                
                // Use arithmetic instead of branching where possible
                const bool valid = (l_in_nom % stride == 0) && (l_in >= 0) && (l_in < L_in);
                const float x_val = valid ? x[x_offset + l_in] : 0.0f;
                const float w_val = weight[w_offset + k_w];
                
                value += x_val * w_val;
            }
        }
        
        // Write output
        if (idx < N * C_out * L_out) {
            y[idx] = value;
        }
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
    
    // Configure kernel launch parameters
    const int threads_per_block = 256; // Multiple of WARP_SIZE
    const int total_elements = N * C_out * L_out;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
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