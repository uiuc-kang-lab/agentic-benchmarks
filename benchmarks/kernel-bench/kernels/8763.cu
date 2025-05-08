#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float compute_contribution(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int x_base, int w_base,
    int l_out, int k_w,
    int stride, int padding, int dilation,
    int L_in) {
    
    int l_in_nom = l_out + padding - k_w * dilation;
    if (l_in_nom % stride != 0) return 0.0f;
    
    int l_in = l_in_nom / stride;
    if (l_in < 0 || l_in >= L_in) return 0.0f;
    
    return x[x_base + l_in] * weight[w_base + k_w];
}

__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation,
    int n_start, int n_end) {
    
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_w = &shared_mem[blockDim.x];
    
    int chunk_N = n_end - n_start;
    int total_elements = chunk_N * C_out * L_out;
    int tid = threadIdx.x;
    
    for (int idx = blockIdx.x * blockDim.x + tid; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        int l_out = idx % L_out;
        int c_out = (idx / L_out) % C_out;
        int n_index = idx / (L_out * C_out);
        int n = n_start + n_index;
        
        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int x_base = n * C_in * L_in + c_in * L_in;
            int w_base = c_in * C_out * K_w + c_out * K_w;
            
            if (tid < K_w) {
                shared_w[tid] = weight[w_base + tid];
            }
            __syncthreads();
            
            for (int k_w = 0; k_w < K_w; ++k_w) {
                sum += compute_contribution(
                    x, shared_w,
                    x_base, 0,
                    l_out, k_w,
                    stride, padding, dilation, L_in);
            }
            __syncthreads();
        }
        y[n * C_out * L_out + c_out * L_out + l_out] = sum;
    }
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
    
    int threads = 256;
    int shared_mem_size = threads * sizeof(float) + K_w * sizeof(float);
    
    int n_streams = 2;
    int chunk_size = (N + n_streams - 1) / n_streams;
    
    cudaStream_t streams[2];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    for (int i = 0; i < n_streams; i++) {
        int n_start = i * chunk_size;
        int n_end = std::min(n_start + chunk_size, N);
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