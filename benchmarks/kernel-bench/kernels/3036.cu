#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_FEATURES 2048

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];
    float* warp_max = smem;
    float* warp_sum = smem + num_warps;
    float* temp_storage = smem + 2*num_warps;

    int items_per_warp = (num_features + num_warps - 1) / num_warps;
    int warp_start = warp_id * items_per_warp;
    int warp_end = min(warp_start + items_per_warp, num_features);

    float thread_max = -INFINITY;
    
    if (num_features <= MAX_FEATURES) {
        for (int i = tid; i < num_features; i += blockDim.x) {
            temp_storage[i] = x[batch_idx * num_features + i];
        }
        __syncthreads();
        
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            thread_max = max(thread_max, temp_storage[i]);
        }
    } else {
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            thread_max = max(thread_max, x[batch_idx * num_features + i]);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) warp_max[warp_id] = thread_max;
    __syncthreads();

    if (tid == 0) {
        float block_max = warp_max[0];
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();
    float max_val = warp_max[0];

    float thread_sum = 0.0f;
    
    if (num_features <= MAX_FEATURES) {
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            float val = __expf(temp_storage[i] - max_val);
            temp_storage[i] = val;
            thread_sum += val;
        }
    } else {
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            float val = __expf(x[batch_idx * num_features + i] - max_val);
            y[batch_idx * num_features + i] = val;
            thread_sum += val;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) warp_sum[warp_id] = thread_sum;
    __syncthreads();

    if (tid == 0) {
        float block_sum = warp_sum[0];
        for (int i = 1; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum;
    }
    __syncthreads();
    float sum_val = warp_sum[0];

    if (num_features <= MAX_FEATURES) {
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            y[batch_idx * num_features + i] = temp_storage[i] / sum_val;
        }
    } else {
        for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
            y[batch_idx * num_features + i] /= sum_val;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    size_t shared_mem_size;
    
    if (num_features <= MAX_FEATURES) {
        shared_mem_size = (2 * num_warps + num_features) * sizeof(float);
    } else {
        shared_mem_size = 2 * num_warps * sizeof(float);
    }
    
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