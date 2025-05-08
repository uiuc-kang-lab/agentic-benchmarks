#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, 
                             const int batch_size, const int num_features) {
    // 2D block and thread indexing
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for max and sum reduction
    __shared__ float s_max[BLOCK_DIM_Y];
    __shared__ float s_sum[BLOCK_DIM_Y];
    
    if (batch_idx >= batch_size) return;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        s_max[threadIdx.y] = -INFINITY;
        s_sum[threadIdx.y] = 0.0f;
    }
    __syncthreads();
    
    // First pass: find max value
    if (feature_idx < num_features) {
        float val = x[batch_idx * num_features + feature_idx];
        atomicMax((int*)&s_max[threadIdx.y], __float_as_int(val));
    }
    __syncthreads();
    
    // Compute local exponentials and sum
    float local_sum = 0.0f;
    if (feature_idx < num_features) {
        float val = x[batch_idx * num_features + feature_idx];
        float exp_val = __expf(val - s_max[threadIdx.y]);
        y[batch_idx * num_features + feature_idx] = exp_val;
        local_sum = exp_val;
    }
    
    // Sum reduction within block
    atomicAdd(&s_sum[threadIdx.y], local_sum);
    __syncthreads();
    
    // Final normalization
    if (feature_idx < num_features) {
        y[batch_idx * num_features + feature_idx] /= s_sum[threadIdx.y];
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks(
        (num_features + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (batch_size + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
    
    softmax_kernel<<<blocks, threads>>>(x, y, batch_size, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x.size(0),
        x.size(1)
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}