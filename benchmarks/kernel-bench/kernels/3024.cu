#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Warp-level reduction for maximum
__inline__ __device__ float warpReduceMax(float val) {
    // Use shuffle-based reduction for maximum within a warp
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__inline__ __device__ float warpReduceSum(float val) {
    // Use shuffle-based reduction for summation within a warp
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized softmax kernel with improved memory coalescing and warp-level reductions
__global__ void softmax_kernel_optimized(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;  // one row per block
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    // Determine warp id and lane within the warp
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Pointers to the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Step 1: Compute the maximum value in the row
    float thread_max = -INFINITY;
    for (int j = tid; j < num_features; j += block_threads) {
        float val = x_row[j];
        thread_max = (val > thread_max) ? val : thread_max;
    }
    // Intra-warp reduction
    float warp_max = warpReduceMax(thread_max);

    // Shared memory to store each warp's max
    __shared__ float sdata_max[THREADS_PER_BLOCK / WARP_SIZE];
    if (lane == 0) {
        sdata_max[warp_id] = warp_max;
    }
    __syncthreads();

    // Final reduction: let the first warp compute the final max
    float row_max = -INFINITY;
    if (tid < block_threads / WARP_SIZE) {
        row_max = sdata_max[tid];
    }
    if (warp_id == 0) {
        row_max = warpReduceMax(row_max);
    }
    // Broadcast the computed maximum to all threads
    row_max = __shfl_sync(0xffffffff, row_max, 0);
    __syncthreads();

    // Step 2: Compute exponentials and partial sums
    float thread_sum = 0.0f;
    for (int j = tid; j < num_features; j += block_threads) {
        float exp_val = __expf(x_row[j] - row_max);
        y_row[j] = exp_val;  // temporary storage for exponentials
        thread_sum += exp_val;
    }
    // Intra-warp summation
    float warp_sum = warpReduceSum(thread_sum);

    // Shared memory to store each warp's sum
    __shared__ float sdata_sum[THREADS_PER_BLOCK / WARP_SIZE];
    if (lane == 0) {
        sdata_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction of sums
    float row_sum = 0.0f;
    if (tid < block_threads / WARP_SIZE) {
        row_sum = sdata_sum[tid];
    }
    if (warp_id == 0) {
        row_sum = warpReduceSum(row_sum);
    }
    // Broadcast the final sum to all threads
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    __syncthreads();

    // Step 3: Normalize the exponentials to obtain the final softmax values
    for (int j = tid; j < num_features; j += block_threads) {
        y_row[j] = y_row[j] / row_sum; 
    }
}

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    // Launch the optimized kernel
    softmax_kernel_optimized<<<grid_dim, block_dim>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ forward function (pybind11 interface)
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

// pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softmax forward (CUDA)");
}
