#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];
    float* warp_results = smem;

    const int items_per_warp = (num_features + num_warps - 1) / num_warps;
    const int warp_start = wid * items_per_warp;
    const int warp_end = min(warp_start + items_per_warp, num_features);

    float thread_max = -INFINITY;
    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        thread_max = max(thread_max, x[batch_idx * num_features + idx]);
    }
    
    float warp_max = warp_reduce_max(thread_max);

    if (lane == 0) {
        warp_results[wid] = warp_max;
    }
    
    __syncthreads();

    float global_max;
    if (tid == 0) {
        global_max = warp_results[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = max(global_max, warp_results[i]);
        }
        warp_results[0] = global_max;
    }
    __syncthreads();
    
    global_max = warp_results[0];

    float thread_sum = 0.0f;
    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        const int global_idx = batch_idx * num_features + idx;
        float val = __expf(x[global_idx] - global_max);
        y[global_idx] = val;
        thread_sum += val;
    }

    float warp_sum = warp_reduce_sum(thread_sum);

    if (lane == 0) {
        warp_results[wid] = warp_sum;
    }

    __syncthreads();

    if (tid == 0) {
        float global_sum = warp_results[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += warp_results[i];
        }
        warp_results[0] = global_sum;
    }
    __syncthreads();

    float global_sum = warp_results[0];

    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        y[batch_idx * num_features + idx] /= global_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    int shared_mem_size = (THREADS_PER_BLOCK / WARP_SIZE) * sizeof(float);
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
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