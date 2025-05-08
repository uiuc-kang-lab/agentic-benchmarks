#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256

// CUDA kernel for softmax using stride loops and proper boundary handling
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row of the input (batch element)
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Pointers to the start of the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Allocate shared memory: first half for max reduction, second half for sum reduction
    extern __shared__ float shared_mem[];
    float* max_shared = shared_mem;
    float* sum_shared = shared_mem + blockSize;

    // Step 1: Compute the maximum value in the row using stride loop
    float max_val_local = -INFINITY;
    for (int i = tid; i < num_features; i += blockSize) {
        float val = x_row[i];
        if (val > max_val_local) {
            max_val_local = val;
        }
    }
    max_shared[tid] = max_val_local;
    __syncthreads();

    // Reduction to compute the maximum value in shared memory
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (max_shared[tid + s] > max_shared[tid]) {
                max_shared[tid] = max_shared[tid + s];
            }
        }
        __syncthreads();
    }

    float row_max = max_shared[0];
    __syncthreads();

    // Step 2: Compute exponentials and partial sum using stride loop
    float sum_val_local = 0.0f;
    for (int i = tid; i < num_features; i += blockSize) {
        float exp_val = __expf(x_row[i] - row_max);
        y_row[i] = exp_val;  // Store the intermediate exponential result
        sum_val_local += exp_val;
    }
    sum_shared[tid] = sum_val_local;
    __syncthreads();

    // Reduction to compute the sum of exponentials
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        __syncthreads();
    }

    float total_sum = sum_shared[0];
    __syncthreads();

    // Step 3: Normalize the results
    for (int i = tid; i < num_features; i += blockSize) {
        y_row[i] = y_row[i] / total_sum;
    }
}

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);

    // Allocate shared memory for both max and sum reductions
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK * 2;

    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
}

// C++ forward function exposed to PyTorch
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

// Pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
