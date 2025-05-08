#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    extern __shared__ float smem[];
    float* warp_max = smem;
    float* warp_sum = smem + blockDim.x / WARP_SIZE;

    int items_per_warp = (num_features + blockDim.x - 1) / blockDim.x;
    int start_idx = tid * items_per_warp;
    int end_idx = min(start_idx + items_per_warp, num_features);

    float thread_max = -INFINITY;
    for (int i = start_idx; i < end_idx; i++) {
        float val = x[batch_idx * num_features + i];
        thread_max = max(thread_max, val);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) warp_max[warp_id] = thread_max;
    __syncthreads();

    if (tid == 0) {
        float block_max = warp_max[0];
        for (int i = 1; i < blockDim.x / WARP_SIZE; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();
    float max_val = warp_max[0];

    float thread_sum = 0.0f;
    for (int i = start_idx; i < end_idx; i++) {
        float val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = val;
        thread_sum += val;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) warp_sum[warp_id] = thread_sum;
    __syncthreads();

    if (tid == 0) {
        float block_sum = warp_sum[0];
        for (int i = 1; i < blockDim.x / WARP_SIZE; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum;
    }
    __syncthreads();
    float sum_val = warp_sum[0];

    for (int i = start_idx; i < end_idx; i++) {
        y[batch_idx * num_features + i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    size_t shared_mem_size = 2 * (block.x / WARP_SIZE) * sizeof(float);
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