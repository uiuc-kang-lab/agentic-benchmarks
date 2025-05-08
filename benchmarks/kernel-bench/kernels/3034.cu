#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_optimized_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row of the input tensor
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Partition the row among warps
    int elementsPerWarp = (num_features + num_warps - 1) / num_warps;
    int start = warp_id * elementsPerWarp;
    int end = min(start + elementsPerWarp, num_features);

    // Shared memory for reduction, using two sets for storing max and sum values
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + num_warps;

    // Compute local maximum and reduction for max in warp using shuffle
    float local_max = -INFINITY;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        float val = x[batch_idx * num_features + i];
        local_max = fmaxf(local_max, val);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) {
        smax[warp_id] = local_max;
    }
    __syncthreads();

    // Max reduction among all warps
    if (tid == 0) {
        float global_max = -INFINITY;
        for (int i = 0; i < num_warps; i++) {
            global_max = fmaxf(global_max, smax[i]);
        }
        smax[0] = global_max;
    }
    __syncthreads();
    float global_max = smax[0];

    // Compute local sum of exponentials
    float local_sum = 0.0f;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        float val = x[batch_idx * num_features + i];
        float exp_val = __expf(val - global_max);
        y[batch_idx * num_features + i] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduction for sums using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    // Sum reduction among warps
    if (tid == 0) {
        float global_sum = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            global_sum += ssum[i];
        }
        ssum[0] = global_sum;
    }
    __syncthreads();
    float global_sum = ssum[0];

    // Normalize using coalesced access within each warp's segment
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        y[batch_idx * num_features + i] = y[batch_idx * num_features + i] / global_sum;
    }
}

// Host function to launch the optimized softmax kernel
void softmax_optimized_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * 2 * num_warps;
    softmax_optimized_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_optimized_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

torch::Tensor optimized_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_optimized_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_forward", &optimized_forward, "Softmax optimized forward (CUDA)");
}