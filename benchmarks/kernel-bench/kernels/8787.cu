#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ConvTranspose1D with stride loop handling
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {

    // Shared memory for weight tiles
    extern __shared__ float shared_weight[];
    
    const int TILE_SIZE = 32;  // Adjust based on your GPU's shared memory size
    int total_elements = N * C_out * L_out;
    
    // Each thread processes multiple output elements using a stride loop
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += blockDim.x * gridDim.x) {

        int l_out = index % L_out;
        int c_out = (index / L_out) % C_out;
        int n = index / (L_out * C_out);

        float value = bias != nullptr ? bias[c_out] : 0.0f;
        int x_base = n * C_in * L_in;
        
        // Process input channels in tiles
        for (int c_in_base = 0; c_in_base < C_in; c_in_base += TILE_SIZE) {
            // Load weight tile into shared memory
            __syncthreads();
            int tid = threadIdx.x;
            int remaining_channels = min(TILE_SIZE, C_in - c_in_base);
            
            // Collaborative loading of weights into shared memory
            for (int i = tid; i < remaining_channels * K_w; i += blockDim.x) {
                int local_c_in = i / K_w;
                int local_k_w = i % K_w;
                int weight_idx = (c_in_base + local_c_in) * C_out * K_w + c_out * K_w + local_k_w;
                shared_weight[i] = weight[weight_idx];
            }
            __syncthreads();
            
            // Process the tile
            for (int local_c_in = 0; local_c_in < remaining_channels; ++local_c_in) {
                int c_in = c_in_base + local_c_in;
                int x_offset = x_base + c_in * L_in;
                
                #pragma unroll 4
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int l_in_nom = l_out + padding - k_w * dilation;
                    if (l_in_nom % stride == 0) {
                        int l_in = l_in_nom / stride;
                        if (l_in >= 0 && l_in < L_in) {
                            value += x[x_offset + l_in] * shared_weight[local_c_in * K_w + k_w];
                        }
                    }
                }
            }
        }
        
        y[index] = value;
    }
}

// Forward function that interfaces with PyTorch and launches the kernel
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
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &conv_transpose1d_forward, "Conv Transpose1D forward (CUDA) with stride loop for extended workload handling",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("dilation") = 1);
}
