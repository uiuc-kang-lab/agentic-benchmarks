#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float smem[];
    float* smax = smem;              // size: num_warps
    float* ssum = smem + num_warps;  // size: num_warps

    // First compute max using warp-level parallelism for efficiency
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        local_max = max(local_max, x_row[i]);
    }

    // Warp-level reduction using shuffle
    uint32_t mask = 0xffffffff;
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
    }

    if (lane == 0) {
        smax[warp_id] = local_max;
    }
    __syncthreads();

    // Block-level reduction for final max
    if (tid == 0) {
        float global_max = smax[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = max(global_max, smax[i]);
        }
        smax[0] = global_max;
    }
    __syncthreads();
    float max_val = smax[0];

    // Compute exponentials and local sum using coalesced memory access
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level reduction for sum using shuffle
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if (lane == 0) {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    // Block-level reduction for final sum
    if (tid == 0) {
        float global_sum = ssum[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += ssum[i];
        }
        ssum[0] = global_sum;
    }
    __syncthreads();
    float sum_val = ssum[0];

    // Normalize with coalesced memory access
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    int shared_mem_size = sizeof(float) * 2 * num_warps;
    
    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}