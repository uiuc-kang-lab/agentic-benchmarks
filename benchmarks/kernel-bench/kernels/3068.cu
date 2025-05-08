#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    extern __shared__ float sdata[];
    float* max_shared = sdata;
    float* sum_shared = &sdata[stride];

    // Find max value using thread-local variable first
    float thread_max = -INFINITY;
    // Process elements in chunks for coalesced memory access
    for (int chunk_start = 0; chunk_start < num_features; chunk_start += stride) {
        int i = chunk_start + tid;
        if (i < num_features) {
            thread_max = max(thread_max, x_row[i]);
        }
    }

    // Store thread-local max in shared memory
    max_shared[tid] = thread_max;
    __syncthreads();

    // Parallel reduction for max
    for (int offset = stride/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            max_shared[tid] = max(max_shared[tid], max_shared[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = max_shared[0];
    __syncthreads();

    // Compute exponentials and partial sums in thread-local variable
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;  // Store intermediate result
        thread_sum += exp_val;
    }

    // Store thread-local sum
    sum_shared[tid] = thread_sum;
    __syncthreads();

    // Parallel reduction for sum
    for (int offset = stride/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sum_shared[tid] += sum_shared[tid + offset];
        }
        __syncthreads();
    }

    float sum_val = sum_shared[0];
    float inv_sum = 1.0f / sum_val;

    // Final normalization without atomic operations
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] *= inv_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);

    // Double the shared memory to store both max and sum values
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK * 2;

    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
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