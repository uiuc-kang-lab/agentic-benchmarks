#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define UNROLL_FACTOR 4

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float shared_mem[];
    float* warp_max = shared_mem;
    float* warp_sum = shared_mem + num_warps;

    // Calculate work distribution
    int items_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    int start_idx = tid * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, num_features);

    // Find max value with manual loop unrolling
    float thread_max = -INFINITY;
    int i = start_idx;
    
    #pragma unroll
    for (; i + UNROLL_FACTOR <= end_idx; i += UNROLL_FACTOR) {
        float val0 = x[batch_idx * num_features + i];
        float val1 = x[batch_idx * num_features + i + 1];
        float val2 = x[batch_idx * num_features + i + 2];
        float val3 = x[batch_idx * num_features + i + 3];
        
        thread_max = max(thread_max, val0);
        thread_max = max(thread_max, val1);
        thread_max = max(thread_max, val2);
        thread_max = max(thread_max, val3);
    }
    
    // Handle remaining elements
    for (; i < end_idx; i++) {
        thread_max = max(thread_max, x[batch_idx * num_features + i]);
    }

    // Warp-level reduction unrolled
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Block-level reduction unrolled for warps
    if (tid == 0) {
        float block_max = warp_max[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();
    
    float max_val = warp_max[0];

    // Compute exponentials and sum with manual unrolling
    float thread_sum = 0.0f;
    i = start_idx;
    
    #pragma unroll
    for (; i + UNROLL_FACTOR <= end_idx; i += UNROLL_FACTOR) {
        float val0 = __expf(x[batch_idx * num_features + i] - max_val);
        float val1 = __expf(x[batch_idx * num_features + i + 1] - max_val);
        float val2 = __expf(x[batch_idx * num_features + i + 2] - max_val);
        float val3 = __expf(x[batch_idx * num_features + i + 3] - max_val);
        
        y[batch_idx * num_features + i] = val0;
        y[batch_idx * num_features + i + 1] = val1;
        y[batch_idx * num_features + i + 2] = val2;
        y[batch_idx * num_features + i + 3] = val3;
        
        thread_sum += val0 + val1 + val2 + val3;
    }
    
    // Handle remaining elements
    for (; i < end_idx; i++) {
        float val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = val;
        thread_sum += val;
    }

    // Warp-level sum reduction unrolled
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block-level sum reduction unrolled
    if (tid == 0) {
        float block_sum = warp_sum[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum;
    }
    __syncthreads();
    
    float sum_val = warp_sum[0];

    // Normalize with manual unrolling
    i = start_idx;
    #pragma unroll
    for (; i + UNROLL_FACTOR <= end_idx; i += UNROLL_FACTOR) {
        y[batch_idx * num_features + i] /= sum_val;
        y[batch_idx * num_features + i + 1] /= sum_val;
        y[batch_idx * num_features + i + 2] /= sum_val;
        y[batch_idx * num_features + i + 3] /= sum_val;
    }
    
    for (; i < end_idx; i++) {
        y[batch_idx * num_features + i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    size_t shared_mem_size = 2 * num_warps * sizeof(float);
    
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