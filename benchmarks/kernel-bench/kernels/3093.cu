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
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float shared_data[];
    float* warp_maxes = shared_data;
    float* warp_sums = shared_data + num_warps;

    float thread_max = -INFINITY;
    
    // Ensure coalesced memory access by having consecutive threads access consecutive memory locations
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, x_row[i]);
    }

    float warp_max = warp_reduce_max(thread_max);
    
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) {
            warp_maxes[0] = val;
        }
    }
    __syncthreads();

    const float row_max = warp_maxes[0];
    float thread_sum = 0.0f;

    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        float exp_val = __expf(x_row[i] - row_max);
        y_row[i] = exp_val; // store intermediate result
        thread_sum += exp_val;
    }

    float warp_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float inv_sum = 1.0f / warp_sums[0];

    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        y_row[i] *= inv_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * num_warps * 2;

    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
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
