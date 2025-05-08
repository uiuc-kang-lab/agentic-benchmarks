#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__device__ inline int compute_l_in(int l_out, int k, int stride, int padding, int dilation, int L_in) {
    int l_in_nom = l_out + padding - k * dilation;
    if (l_in_nom % stride != 0) return -1;
    int l_in = l_in_nom / stride;
    return (l_in >= 0 && l_in < L_in) ? l_in : -1;
}

__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out, int K_w,
    int stride, int padding, int dilation) {
    
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int l_out = blockIdx.x * blockDim.x + tx;
    int c_out = blockIdx.y * blockDim.y + ty;
    
    if (l_out >= L_out || c_out >= C_out) return;
    
    for (int n = 0; n < N; n++) {
        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
        
        for (int c_in_tile = 0; c_in_tile < C_in; c_in_tile += TILE_SIZE) {
            for (int k = 0; k < K_w; k += TILE_SIZE) {
                if ((ty + c_in_tile) < C_in && (tx + k) < K_w) {
                    s_weight[ty][tx] = weight[(ty + c_in_tile) * C_out * K_w + c_out * K_w + (tx + k)];
                } else {
                    s_weight[ty][tx] = 0.0f;
                }
            }
            
            int l_in = compute_l_in(l_out, tx, stride, padding, dilation, L_in);
            if (l_in != -1 && (ty + c_in_tile) < C_in) {
                s_input[ty][tx] = x[n * C_in * L_in + (ty + c_in_tile) * L_in + l_in];
            } else {
                s_input[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; k++) {
                if ((c_in_tile + k) < C_in) {
                    sum += s_input[k][tx] * s_weight[k][ty];
                }
            }
            
            __syncthreads();
        }
        
        if (l_out < L_out && c_out < C_out) {
            y[n * C_out * L_out + c_out * L_out + l_out] = sum;
        }
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
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((L_out + TILE_SIZE - 1) / TILE_SIZE,
                (C_out + TILE_SIZE - 1) / TILE_SIZE);
    
    conv_transpose1d_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K_w,
        stride, padding, dilation);
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
    return y;
}