#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_SHARED_SIZE 2048

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {
    
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int total_elements = N * C_out * L_out;
    if (index >= total_elements) return;
    
    int l_out = index % L_out;
    int c_out = (index / L_out) % C_out;
    int n = index / (L_out * C_out);
    
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Compute partial sums in shared memory
    float local_sum = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in += BLOCK_SIZE) {
        int remaining = min(BLOCK_SIZE, C_in - c_in);
        
        if (tid < remaining) {
            for (int k_w = 0; k_w < K_w; ++k_w) {
                int l_in_nom = l_out + padding - k_w * dilation;
                if (l_in_nom % stride == 0) {
                    int l_in = l_in_nom / stride;
                    if (l_in >= 0 && l_in < L_in) {
                        float x_val = x[n * C_in * L_in + (c_in + tid) * L_in + l_in];
                        float w_val = weight[(c_in + tid) * C_out * K_w + c_out * K_w + k_w];
                        local_sum += x_val * w_val;
                    }
                }
            }
        }
        
        shared_mem[tid] = local_sum;
        __syncthreads();
        
        // Warp-level reduction
        if (tid < WARP_SIZE) {
            for (int i = tid + WARP_SIZE; i < remaining; i += WARP_SIZE) {
                local_sum += shared_mem[i];
            }
        }
        
        // Final warp reduction using shuffle
        if (tid < WARP_SIZE) {
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            }
        }
        
        if (tid == 0) {
            value += local_sum;
        }
        __syncthreads();
    }
    
    if (index < total_elements) {
        y[n * C_out * L_out + c_out * L_out + l_out] = value;
    }
}

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
    
    int total_elements = N * C_out * L_out;
    int threads = BLOCK_SIZE;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose1d_kernel<<<blocks, threads, BLOCK_SIZE * sizeof(float)>>>(
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