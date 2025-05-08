#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

// CUDA kernel for softmax using stride loops to handle large workloads and correct boundary handling
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row (batch element)
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pointers for the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Allocate shared memory: first part for max reduction, second for sum reduction
    extern __shared__ float shared[];
    float* s_max = shared;            // size: blockDim.x
    float* s_sum = shared + stride;     // size: blockDim.x

    // Step 1: compute local maximum using stride loop
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        local_max = fmaxf(local_max, val);
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Warp-based reduction for maximum
    unsigned mask = 0xffffffff;
    // First, perform warp-level reduction
    float thread_max = local_max;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, offset));
    }
    // Write each warp's maximum to shared memory
    if (tid % warpSize == 0) {
        s_max[tid / warpSize] = thread_max;
    }
    __syncthreads();
    // Final reduction over warp leaders
    if (tid < blockDim.x / warpSize) {
        thread_max = s_max[tid];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, offset));
        }
        s_max[tid] = thread_max;
    }
    __syncthreads();
    float max_val = s_max[0];


    // Step 2: compute exponentials and a partial sum using stride loop
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        // Compute exponential in a numerically stable way
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val; // store intermediate result
        local_sum += exp_val;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduction to compute the total sum
    for (int s = stride >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float total_sum = s_sum[0];
    __syncthreads();

    // Step 3: normalize the results to get softmax output
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / total_sum;
    }
}

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    
    // Shared memory allocation: two arrays of size THREADS_PER_BLOCK (one for max and one for sum)
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
