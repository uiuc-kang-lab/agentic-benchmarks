#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    extern __shared__ float shared[];
    float* warp_maxes = shared;
    float* warp_sums = &shared[num_warps];

    // Find max value using thread-local variable first
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, x_row[i]);
    }

    // Warp-level reduction for max
    float warp_max = warp_reduce_max(thread_max);

    // Store warp results in shared memory
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0 && lane_id < num_warps) {
        float final_max = warp_maxes[lane_id];
        final_max = warp_reduce_max(final_max);
        warp_maxes[0] = final_max;
    }
    __syncthreads();

    float max_val = warp_maxes[0];

    // Compute exponentials and partial sums
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        thread_sum += exp_val;
    }

    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(thread_sum);

    // Store warp results
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0 && lane_id < num_warps) {
        float final_sum = warp_sums[lane_id];
        final_sum = warp_reduce_sum(final_sum);
        warp_sums[0] = final_sum;
    }
    __syncthreads();

    float sum_val = warp_sums[0];
    float inv_sum = 1.0f / sum_val;

    // Final normalization
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        y_row[i] *= inv_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);

    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * num_warps * 2;

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