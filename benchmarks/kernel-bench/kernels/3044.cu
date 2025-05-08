#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)

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
    const int base_idx = batch_idx * num_features;
    
    const int elements_per_warp = (num_features + NUM_WARPS - 1) / NUM_WARPS;
    const int warp_start = wid * elements_per_warp;
    const int warp_end = min(warp_start + elements_per_warp, num_features);

    float thread_max = -INFINITY;
    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        thread_max = max(thread_max, x[base_idx + idx]);
    }
    
    float warp_max = warp_reduce_max(thread_max);
    
    __shared__ float smem[NUM_WARPS];
    if (lane == 0) {
        smem[wid] = warp_max;
    }
    __syncthreads();

    float global_max;
    if (wid == 0) {
        float val = (lane < NUM_WARPS) ? smem[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    global_max = smem[0];

    float thread_sum = 0.0f;
    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        float val = __expf(x[base_idx + idx] - global_max);
        y[base_idx + idx] = val;
        thread_sum += val;
    }
    
    float warp_sum = warp_reduce_sum(thread_sum);
    
    if (lane == 0) {
        smem[wid] = warp_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();
    float global_sum = smem[0];

    for (int idx = warp_start + lane; idx < warp_end; idx += WARP_SIZE) {
        y[base_idx + idx] /= global_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    softmax_kernel<<<grid, block>>>(x, y, num_features);
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