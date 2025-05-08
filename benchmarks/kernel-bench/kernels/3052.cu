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

    // Point to current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float smem[];
    float* smax = smem;              // size: num_warps
    float* ssum = smem + num_warps;  // size: num_warps

    // First pass: find max using warp-level parallelism for small/medium features
    // and block-level for large features
    float local_max = -INFINITY;
    if (num_features <= 512) {
        // Warp-based approach for better efficiency on smaller sizes
        int elementsPerWarp = (num_features + num_warps - 1) / num_warps;
        int start = warp_id * elementsPerWarp;
        int end = min(start + elementsPerWarp, num_features);
        
        for (int i = start + lane; i < end; i += WARP_SIZE) {
            local_max = max(local_max, x_row[i]);
        }
        
        // Warp-level reduction using shuffle
        uint32_t mask = 0xffffffff;
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            local_max = max(local_max, __shfl_down_sync(mask, local_max, offset));
        }
        
        if (lane == 0) smax[warp_id] = local_max;
    } else {
        // Block-based approach for better load balancing on larger sizes
        for (int i = tid; i < num_features; i += blockDim.x) {
            local_max = max(local_max, x_row[i]);
        }
        smax[tid] = local_max;
    }
    __syncthreads();

    // Reduce to get global max
    if (tid == 0) {
        float global_max = smax[0];
        for (int i = 1; i < (num_features <= 512 ? num_warps : blockDim.x); i++) {
            global_max = max(global_max, smax[i]);
        }
        smax[0] = global_max;
    }
    __syncthreads();
    float global_max = smax[0];

    // Second pass: compute exp and sum, using the same hybrid approach
    float local_sum = 0.0f;
    if (num_features <= 512) {
        int elementsPerWarp = (num_features + num_warps - 1) / num_warps;
        int start = warp_id * elementsPerWarp;
        int end = min(start + elementsPerWarp, num_features);
        
        for (int i = start + lane; i < end; i += WARP_SIZE) {
            float exp_val = __expf(x_row[i] - global_max);
            y_row[i] = exp_val;
            local_sum += exp_val;
        }
        
        // Warp-level reduction using shuffle
        uint32_t mask = 0xffffffff;
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        
        if (lane == 0) ssum[warp_id] = local_sum;
    } else {
        for (int i = tid; i < num_features; i += blockDim.x) {
            float exp_val = __expf(x_row[i] - global_max);
            y_row[i] = exp_val;
            local_sum += exp_val;
        }
        ssum[tid] = local_sum;
    }
    __syncthreads();

    // Reduce to get global sum
    if (tid == 0) {
        float global_sum = ssum[0];
        for (int i = 1; i < (num_features <= 512 ? num_warps : blockDim.x); i++) {
            global_sum += ssum[i];
        }
        ssum[0] = global_sum;
    }
    __syncthreads();
    float global_sum = ssum[0];

    // Final normalization pass
    if (num_features <= 512) {
        int elementsPerWarp = (num_features + num_warps - 1) / num_warps;
        int start = warp_id * elementsPerWarp;
        int end = min(start + elementsPerWarp, num_features);
        for (int i = start + lane; i < end; i += WARP_SIZE) {
            y_row[i] /= global_sum;
        }
    } else {
        for (int i = tid; i < num_features; i += blockDim.x) {
            y_row[i] /= global_sum;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    int shared_mem_size = sizeof(float) * 2 * THREADS_PER_BLOCK;
    
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