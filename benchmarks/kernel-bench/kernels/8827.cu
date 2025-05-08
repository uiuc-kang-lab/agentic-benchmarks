#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Branchless clamp using CUDA intrinsics
__device__ __forceinline__ int clamp_int(int val, int lower, int upper) {
    // Using built-in min and max
    return max(lower, min(val, upper));
}

// Device function that computes a single output element using shared memory tiling
__device__ float compute_output_element_branchless(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int C_in, int L_in, int K_w,
    int stride, int padding, int dilation,
    int c_out, int l_out, int n, int C_out) {

    // Shared memory for input and weight tiles
    extern __shared__ float shared_mem[];
    float* x_shared = shared_mem;
    float* w_shared = &shared_mem[blockDim.x];  // Second half of shared memory for weights

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Process input in tiles to reduce global memory traffic
    const int TILE_SIZE = 32;  // Adjust based on your specific needs
    for (int tile = 0; tile < C_in; tile += TILE_SIZE) {
        int tile_end = min(tile + TILE_SIZE, C_in);
        
        // Cooperatively load input tile into shared memory
        for (int c_in = tile + threadIdx.x; c_in < tile_end; c_in += blockDim.x) {
            int x_batch_offset = c_in * L_in;
            int l_in_base = l_out + padding;
            int l_in_safe = clamp_int(l_in_base / stride, 0, L_in - 1);
            x_shared[c_in - tile] = x[n * C_in * L_in + x_batch_offset + l_in_safe];
        }
        
        // Load weights for this tile into shared memory
        for (int c_in = tile + threadIdx.x; c_in < tile_end; c_in += blockDim.x) {
            for (int k = 0; k < K_w; ++k) {
                int w_idx = c_in * C_out * K_w + c_out * K_w + k;
                w_shared[(c_in - tile) * K_w + k] = weight[w_idx];
            }
        }
        
        __syncthreads();

        // Process the tile
        for (int c_in = tile; c_in < tile_end; ++c_in) {
            int local_c_in = c_in - tile;
            #pragma unroll
            for (int k_w = 0; k_w < K_w; ++k_w) {
                // Compute the nominal input position
                int r = l_out + padding - k_w * dilation;
                // Compute l_in and the remainder in a branchless manner
                int l_in = r / stride;
                int rem = r - l_in * stride;
                
                // Compute validity mask
                int divisible = (rem == 0);
                int in_range = ((l_in >= 0) && (l_in < L_in));
                int valid = divisible * in_range;
                
                float x_val = x_shared[local_c_in];
                float w_val = w_shared[local_c_in * K_w + k_w];
                value += valid * (x_val * w_val);
            }
        }
        
        __syncthreads();
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
