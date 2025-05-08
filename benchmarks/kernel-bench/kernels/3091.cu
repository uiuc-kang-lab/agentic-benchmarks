#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define block and warp sizes
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Warp-level reduction for maximum using shuffle instructions
__inline__ __device__ float warp_reduce_max(float val) {
    // Use full mask of 32 threads
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val < other) ? other : val;
    }
    return val;
}

// Warp-level reduction for sum using shuffle instructions
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized CUDA kernel combining efficient warp-level reductions and shared memory
// Each block handles one row (batch element) of the input tensor
__global__ void softmax_kernel_opt(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Dynamically allocated shared memory: used for both max and sum reductions across warps
    extern __shared__ float shared_data[];  // size should be at least (THREADS_PER_BLOCK / WARP_SIZE) floats

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int batch_idx = blockIdx.x;

    // Pointers to the current row for input and output
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Step 1: Compute the maximum value in the row for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }
    
    // Reduce within each warp
    float warp_max = warp_reduce_max(local_max);
    // Each warp writes its max to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = warp_max;
    }
    __syncthreads();

    // First warp threads load warp maximums and further reduce them
    float row_max = -INFINITY;
    if (tid < num_warps) {
        row_max = shared_data[tid];
    }
    row_max = warp_reduce_max(row_max);
    if (tid == 0) {
        shared_data[0] = row_max;  // store the global row max
    }
    __syncthreads();
    row_max = shared_data[0];

    // Step 2: Compute exponentials and their sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        // Compute exponentials with max subtraction for stability
        float exp_val = __expf(x_row[i] - row_max);
        y_row[i] = exp_val;  // store temporary result
        local_sum += exp_val;
    }
    
    // Reduce local sums within warp
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        shared_data[warp_id] = warp_sum;
    }
    __syncthreads();

    // First warp reduces the warp sums to get global sum
    float global_sum = 0.0f;
    if (tid < num_warps) {
        global_sum = shared_data[tid];
    }
    global_sum = warp_reduce_sum(global_sum);
    if (tid == 0) {
        shared_data[0] = global_sum;  // store the global sum
    }
    __syncthreads();
    global_sum = shared_data[0];

    // Step 3: Normalize the computed exponentials
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] /= global_sum;
    }
}

// C++ API: Launch the CUDA kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * num_warps;  // Shared memory for warp-level reductions

    softmax_kernel_opt<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    // Optional: check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in softmax_kernel_opt: %s\n", cudaGetErrorString(err));
    }
}

// Pybind11 interface
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto y = torch::empty_like(x);

    softmax_forward_cuda(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        num_features
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA optimized)");
}
