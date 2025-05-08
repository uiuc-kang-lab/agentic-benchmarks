#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_SHARED_FEATURES 1024

__global__ void optimized_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];
    float* warp_max = smem;
    float* warp_sum = smem + num_warps;
    float* shared_data = smem + 2*num_warps;

    // Load data into shared memory if feasible
    bool use_shared = (num_features <= MAX_SHARED_FEATURES);
    if (use_shared) {
        for (int fid = tid; fid < num_features; fid += blockDim.x) {
            shared_data[fid] = x[batch_idx * num_features + fid];
        }
    }

    __syncthreads();

    // Compute maximum value in each warp
    float local_max = -INFINITY;
    for (int fid = warp_id * WARP_SIZE + lane; fid < num_features; fid += WARP_SIZE * num_warps) {
        float val = use_shared ? shared_data[fid] : x[batch_idx * num_features + fid];
        local_max = max(local_max, val);
    }

    // Warp-level reduction to obtain the maximum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    // Compute the block-wide maximum
    if (tid == 0) {
        float block_max = warp_max[0];
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();
    float max_val = warp_max[0];

    // Compute exponentials and sum them
    float local_sum = 0.0f;
    for (int fid = warp_id * WARP_SIZE + lane; fid < num_features; fid += WARP_SIZE * num_warps) {
        float val = use_shared ? shared_data[fid] : x[batch_idx * num_features + fid];
        float exp_val = __expf(val - max_val);
        if (use_shared) shared_data[fid] = exp_val; else y[batch_idx * num_features + fid] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level reduction to compute the sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();

    // Compute the block-wide sum
    if (tid == 0) {
        float block_sum = warp_sum[0];
        for (int i = 1; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum;
    }
    __syncthreads();
    float sum_val = warp_sum[0];

    // Normalize the results
    for (int fid = warp_id * WARP_SIZE + lane; fid < num_features; fid += WARP_SIZE * num_warps) {
        int idx = batch_idx * num_features + fid;
        y[idx] = use_shared ? shared_data[fid] / sum_val : y[idx] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    size_t shared_mem_size = (num_features <= MAX_SHARED_FEATURES) ? (2*num_warps + num_features) * sizeof(float) : 2*num_warps * sizeof(float);
    optimized_softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
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