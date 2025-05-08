#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

__device__ float find_max_value(const float* row, int tid, int stride, int num_features, float* shared_mem) {
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        max_val = max(max_val, row[i]);
    }
    
    shared_mem[tid] = max_val;
    __syncthreads();
    
    for (unsigned int s = stride/2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < stride) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    float result = shared_mem[0];
    __syncthreads();
    return result;
}

__device__ float compute_exp_sum(const float* row, float* out_row, float max_val,
                                int tid, int stride, int num_features, float* shared_mem) {
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(row[i] - max_val);
        out_row[i] = exp_val;
        sum_val += exp_val;
    }
    
    shared_mem[tid] = sum_val;
    __syncthreads();
    
    for (unsigned int s = stride/2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < stride) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    float result = shared_mem[0];
    __syncthreads();
    return result;
}

__device__ void normalize_values(float* row, float sum_val, int tid, int stride, int num_features) {
    for (int i = tid; i < num_features; i += stride) {
        row[i] = row[i] / sum_val;
    }
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float shared_mem[];
    
    float max_val = find_max_value(x_row, tid, stride, num_features, shared_mem);
    float sum_val = compute_exp_sum(x_row, y_row, max_val, tid, stride, num_features, shared_mem);
    normalize_values(y_row, sum_val, tid, stride, num_features);
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
    
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");
    
    int batch_size = x.size(0);
    int num_features = x.size(1);
    
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}