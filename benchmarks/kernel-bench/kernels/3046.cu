#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    extern __shared__ float smem[];
    float* shared_data = smem;
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Initialize shared memory
    for (int i = tid; i < num_features; i += blockDim.x) {
        shared_data[i] = x[batch_idx * num_features + i];
    }
    __syncthreads();

    // First level reduction: find max within each warp
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        thread_max = max(thread_max, shared_data[i]);
    }
    
    // Reduce within warp using shuffle
    float warp_max = warp_reduce_max(thread_max);
    
    // Store warp results
    if (lane == 0) {
        shared_data[wid] = warp_max;
    }
    __syncthreads();

    // Final reduction for max (only first warp)
    if (wid == 0) {
        float block_max = (lane < warps_per_block) ? shared_data[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) {
            shared_data[0] = block_max;
        }
    }
    __syncthreads();
    
    float max_val = shared_data[0];

    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = __expf(shared_data[i] - max_val);
        shared_data[i] = val;  // Store intermediate result
        thread_sum += val;
    }
    __syncthreads();

    // Reduce sum within warp
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // Store warp sums
    if (lane == 0) {
        shared_data[wid] = warp_sum;
    }
    __syncthreads();

    // Final reduction for sum (only first warp)
    if (wid == 0) {
        float block_sum = (lane < warps_per_block) ? shared_data[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared_data[0] = block_sum;
        }
    }
    __syncthreads();
    
    float sum_val = shared_data[0];

    // Compute final softmax values with coalesced writes
    for (int i = tid; i < num_features; i += blockDim.x) {
        y[batch_idx * num_features + i] = shared_data[i] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    // Allocate shared memory for the feature data and reduction results
    size_t shared_mem_size = max(num_features * sizeof(float), 
                                THREADS_PER_BLOCK * sizeof(float));
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

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