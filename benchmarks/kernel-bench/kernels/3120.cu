#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 256

__global__ void softmax_kernel(const float* __restrict__ x, 
                             float* __restrict__ y,
                             const int num_features,
                             const int batch_size,
                             const int rows_per_block) {
    extern __shared__ float smem[];
    float* sdata = smem;
    
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    
    // Calculate which rows this block handles
    const int row_start = blockIdx.x * rows_per_block;
    const int row_end = min(row_start + rows_per_block, batch_size);
    
    // Process multiple rows if needed
    for (int row = row_start; row < row_end; row++) {
        const float* x_row = x + row * num_features;
        float* y_row = y + row * num_features;
        
        // Find max value
        float thread_max = -INFINITY;
        for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
            thread_max = max(thread_max, __ldg(&x_row[i]));
        }
        
        sdata[tid] = thread_max;
        __syncthreads();
        
        // Reduce max within block
        if (tid < 128) sdata[tid] = max(sdata[tid], sdata[tid + 128]);
        __syncthreads();
        if (tid < 64) sdata[tid] = max(sdata[tid], sdata[tid + 64]);
        __syncthreads();
        if (tid < 32) {
            volatile float* vmem = sdata;
            vmem[tid] = max(vmem[tid], vmem[tid + 32]);
            vmem[tid] = max(vmem[tid], vmem[tid + 16]);
            vmem[tid] = max(vmem[tid], vmem[tid + 8]);
            vmem[tid] = max(vmem[tid], vmem[tid + 4]);
            vmem[tid] = max(vmem[tid], vmem[tid + 2]);
            vmem[tid] = max(vmem[tid], vmem[tid + 1]);
        }
        
        const float max_val = sdata[0];
        __syncthreads();
        
        // Compute exp and sum
        float thread_sum = 0.0f;
        const int stride = THREADS_PER_BLOCK;
        #pragma unroll 4
        for (int i = tid; i < num_features; i += stride) {
            const float val = __expf(__ldg(&x_row[i]) - max_val);
            y_row[i] = val;
            thread_sum += val;
        }
        
        // Sum reduction
        sdata[tid] = thread_sum;
        __syncthreads();
        
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
        if (tid < 32) {
            volatile float* vmem = sdata;
            vmem[tid] += vmem[tid + 32];
            vmem[tid] += vmem[tid + 16];
            vmem[tid] += vmem[tid + 8];
            vmem[tid] += vmem[tid + 4];
            vmem[tid] += vmem[tid + 2];
            vmem[tid] += vmem[tid + 1];
        }
        
        const float sum_val = sdata[0];
        __syncthreads();
        
        // Normalize with stride pattern
        #pragma unroll 4
        for (int i = tid; i < num_features; i += stride) {
            y_row[i] /= sum_val;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    const int rows_per_block = max(1, batch_size / MAX_BLOCKS + (batch_size % MAX_BLOCKS != 0));
    const int grid_dim = min(MAX_BLOCKS, (batch_size + rows_per_block - 1) / rows_per_block);
    
    dim3 grid(grid_dim);
    dim3 block(THREADS_PER_BLOCK);
    
    const int smem_size = THREADS_PER_BLOCK * sizeof(float);
    
    softmax_kernel<<<grid, block, smem_size>>>(x, y, num_features, batch_size, rows_per_block);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}