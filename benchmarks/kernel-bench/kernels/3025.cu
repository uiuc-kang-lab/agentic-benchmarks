#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float sdata[];
    
    // Find max using registers first
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        thread_max = max(thread_max, x_row[i]);
    }
    
    // Reduce max values in shared memory
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (unsigned int s = stride/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    
    // No sync needed here since we're only reading max_val
    
    // Compute exponentials and partial sums using registers
    float thread_sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_features; i += stride) {
        float val = __expf(x_row[i] - max_val);
        y_row[i] = val;  // Store intermediate result
        thread_sum += val;
    }
    
    // Reduce sum values in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (unsigned int s = stride/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum_val = sdata[0];
    
    // No sync needed before normalization since we're only reading sum_val
    
    // Normalize with minimal synchronization
    #pragma unroll 4
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
    
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
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