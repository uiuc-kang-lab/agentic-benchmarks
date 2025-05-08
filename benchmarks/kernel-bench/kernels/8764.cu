#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combines stream pipelining with divergence-free arithmetic
__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,       // [N, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K_w]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ y,             // [N, C_out, L_out]
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n_start, int n_end) 
{
    // Shared memory for input and weight data
    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    float* shared_x = &shared_mem[K_w];

    int chunk_N = n_end - n_start;
    int total_elements = chunk_N * C_out * L_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    // Compute output indices
    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n_index = index / (L_out * C_out);
    int n = n_start + n_index;

    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Process input channels in tiles
    const int TILE_SIZE = 32;
    for (int c_in_base = 0; c_in_base < C_in; c_in_base += TILE_SIZE) {
        // Collaborative loading of weight data into shared memory
        if (threadIdx.x < K_w && c_in_base + threadIdx.x/K_w < C_in) {
            shared_weight[threadIdx.x] = weight[(c_in_base + threadIdx.x/K_w) * C_out * K_w + c_out * K_w + (threadIdx.x % K_w)];
        }
        
        __syncthreads();

        // Process this tile of input channels
        int tile_end = min(c_in_base + TILE_SIZE, C_in);
        for (int c_in = c_in_base; c_in < tile_end; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int l_in_nom = l_out + padding - k_w * dilation;
                // Use arithmetic instead of branches
                int mod_valid = 1 - (l_in_nom % stride);
                int l_in = l_in_nom / stride;
                int range_valid = (l_in >= 0 && l_in < L_in) ? 1 : 0;
                int valid = mod_valid & range_valid;
                
                float x_val = valid ? x[x_base + l_in] : 0.0f;
                float w_val = shared_weight[k_w];
                value += x_val * w_val;
            }
        }
        __syncthreads();
    }
    
    y[n * C_out * L_out + c_out * L_out + l_out] = value;
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

    const float* bias_ptr = nullptr;
    if (!bias_obj.is_none()) {
        torch::Tensor bias = bias_obj.cast<torch::Tensor>().contiguous();
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
        bias_ptr = bias.data_ptr<float>();
    }

    int N = x.size(0);
    int C_in = x.size(1);
    int L_in = x.size(2);
    int C_out = weight.size(1);
    int K_w = weight.size(2);
    int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1;

    auto y = torch::empty({N, C_out, L_out}, x.options());

    const int threads = 256;
    const int n_streams = 4;  // Increased number of streams
    const int chunk_size = (N + n_streams - 1) / n_streams;
    
    // Shared memory size calculation
    const int shared_mem_size = (K_w + 32) * sizeof(float);  // For weights and input tile
    
    cudaStream_t streams[4];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    for (int i = 0; i < n_streams; i++) {
        int n_start = i * chunk_size;
        int n_end = min(n_start + chunk_size, N);
        if (n_start >= n_end) break;

        int current_chunk = n_end - n_start;
        int total_elements = current_chunk * C_out * L_out;
        int blocks = (total_elements + threads - 1) / threads;

        conv_transpose1d_kernel_optimized<<<blocks, threads, shared_mem_size, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            y.data_ptr<float>(),
            N, C_in, C_out, L_in, L_out, K_w,
            stride, padding, dilation,
            n_start, n_end);
    }

    for (int i = 0; i < n_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}