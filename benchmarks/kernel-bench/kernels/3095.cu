#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

// This kernel leverages shared memory to cache the entire row of the input tensor.
// It loads the row into shared memory, computes the maximum value, then the exponential values, 
// reduces them to compute the sum, and finally normalizes to produce the softmax output.
// This reduces repeated global memory accesses for the row during the reduction and normalization phases.

__global__ void softmax_kernel_shared(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Dynamically allocated shared memory to cache the entire row
    extern __shared__ float s_data[];  // Size: num_features * sizeof(float)

    // Load the current row from global memory to shared memory in a coalesced manner
    for (int i = tid; i < num_features; i += blockDim.x) {
        s_data[i] = x_row[i];
    }
    __syncthreads();

    // Step 1: Compute the maximum value from the cached row.
    // For simplicity, use thread 0 to perform the reduction over shared memory.
    float row_max = -INFINITY;
    if (tid == 0) {
        for (int i = 0; i < num_features; i++) {
            row_max = fmaxf(row_max, s_data[i]);
        }
    }
    __syncthreads();

    // Broadcast the computed row maximum to all threads using a shared variable
    __shared__ float s_row_max;
    if (tid == 0) {
        s_row_max = row_max;
    }
    __syncthreads();
    row_max = s_row_max;

    // Step 2: Compute exponentials in shared memory using the computed max (in-place update)
    for (int i = tid; i < num_features; i += blockDim.x) {
        s_data[i] = __expf(s_data[i] - row_max);
    }
    __syncthreads();

    // Step 3: Compute the sum of the exponentials; again, let thread 0 do the reduction
    float sum_exp = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < num_features; i++) {
            sum_exp += s_data[i];
        }
    }
    __syncthreads();

    // Broadcast the sum to all threads
    __shared__ float s_sum;
    if (tid == 0) {
        s_sum = sum_exp;
    }
    __syncthreads();
    sum_exp = s_sum;

    // Step 4: Normalize the exponentials and write the result to global memory
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] = s_data[i] / sum_exp;
    }
}

// C++ interface function for the CUDA kernel
// This function verifies input constraints and launches the kernel with appropriate grid and block dimensions
// The shared memory size allocated is exactly num_features * sizeof(float) per block.

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);

    // Choose block size: use min(THREADS_PER_BLOCK, num_features) to ensure full utilization
    int block_size = (num_features < THREADS_PER_BLOCK) ? num_features : THREADS_PER_BLOCK;

    dim3 block_dim(block_size);
    dim3 grid_dim(batch_size);

    // Allocate shared memory to cache one row (num_features floats per block)
    size_t shared_mem_size = num_features * sizeof(float);

    softmax_kernel_shared<<<grid_dim, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_features
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_kernel_shared: %s\n", cudaGetErrorString(err));
    }

    return y;
}

// pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA) using shared memory caching");
}
