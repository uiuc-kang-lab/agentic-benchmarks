#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 128  // Reduced from 256 to 128
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;  // Now 4 warps per block

    extern __shared__ float shared_mem[];
    float* warp_max = shared_mem;
    float* warp_sum = shared_mem + num_warps;

    // Adjust work distribution for smaller block size
    int items_per_thread = (num_features + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int start = tid * items_per_thread;
    int end = min(start + items_per_thread, num_features);

    // First pass: find max
    float thread_max = -INFINITY;
    for (int i = start; i < end; i++) {
        thread_max = max(thread_max, x[batch_idx * num_features + i]);
    }

    // Warp-level reduction for max
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Block-level reduction for max
    if (tid < WARP_SIZE) {
        float block_max = (tid < num_warps) ? warp_max[tid] : -INFINITY;
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            block_max = max(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }
        if (tid == 0) {
            warp_max[0] = block_max;
        }
    }
    __syncthreads();

    float max_val = warp_max[0];

    // Second pass: compute exp and sum
    float thread_sum = 0.0f;
    for (int i = start; i < end; i++) {
        float val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = val;
        thread_sum += val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block-level reduction for sum
    if (tid < WARP_SIZE) {
        float block_sum = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (tid == 0) {
            warp_sum[0] = block_sum;
        }
    }
    __syncthreads();

    float sum_val = warp_sum[0];

    // Final pass: normalize
    for (int i = start; i < end; i++) {
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