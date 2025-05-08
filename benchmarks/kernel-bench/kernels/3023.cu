#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    // Use registers for per-thread max and sum
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // Each thread processes multiple elements strided by blockDim.x
    for (int i = tid; i < num_features; i += blockDim.x) {
        thread_max = max(thread_max, x_row[i]);
    }
    
    // Warp-level reduction for max using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_max = max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }
    
    // Share max across warps using shared memory
    __shared__ float warp_max[8]; // Assuming max 8 warps per block
    
    if (lane_id == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // First thread in block finds global max
    if (tid == 0) {
        float global_max = warp_max[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = max(global_max, warp_max[i]);
        }
        warp_max[0] = global_max;
    }
    __syncthreads();
    
    float max_val = warp_max[0];
    
    // Compute exponentials and sum
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }
    
    // Share sums across warps
    __shared__ float warp_sum[8];
    
    if (lane_id == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First thread in block computes global sum
    if (tid == 0) {
        float global_sum = warp_sum[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += warp_sum[i];
        }
        warp_sum[0] = global_sum;
    }
    __syncthreads();
    
    float sum_val = warp_sum[0];
    
    // Normalize with coalesced memory access
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] = y_row[i] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    softmax_kernel<<<grid_dim, block_dim>>>(x, y, num_features);
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