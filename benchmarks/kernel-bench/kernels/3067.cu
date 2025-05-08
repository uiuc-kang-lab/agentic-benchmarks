#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#include <cfloat>

// Removed warp-level reduction functions in favor of a standard block-level reduction using shared memory.

// CUDA kernel for Softmax with warp-level reductions for balanced workload distribution
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;  // total threads in block

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Phase 1: Compute the maximum value for the row
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        local_max = max(local_max, x_row[i]);
    }

    // Perform warp-level reduction for max
    float max_val = warpReduceMax(local_max);

    // Use shared memory to combine results from each warp
    __shared__ float shared_max[THREADS_PER_BLOCK / 32];
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    if (tid < (blockDim.x / 32)) {
        max_val = shared_max[tid];
        max_val = warpReduceMax(max_val);
        if (tid == 0) {
            shared_max[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared_max[0];

    // Phase 2: Compute exponentials and partial sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level reduction for sum
    float sum_val = warpReduceSum(local_sum);

    // Combine per-warp sums using shared memory
    __shared__ float shared_sum[THREADS_PER_BLOCK / 32];
    if (lane == 0) {
        shared_sum[warp_id] = sum_val;
    }
    __syncthreads();

    if (tid < (blockDim.x / 32)) {
        sum_val = shared_sum[tid];
        sum_val = warpReduceSum(sum_val);
        if (tid == 0) {
            shared_sum[0] = sum_val;
        }
    }
    __syncthreads();
    sum_val = shared_sum[0];

    // Phase 3: Normalize exponentials
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);

    // Shared memory is not required as we use warp shuffle intrinsics
    softmax_kernel<<<grid_dim, block_dim>>>(x, y, num_features);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ forward function
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

// pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA) with balanced workload distribution");
}
