#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define FEATURES_THRESHOLD 512  // Threshold to switch between implementations

__global__ void warp_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, 
                                  int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory for both implementations
    extern __shared__ float smem[];
    float* sdata = smem;

    if (num_features <= FEATURES_THRESHOLD) {
        // Use stride-based approach for smaller feature counts
        float local_max = -INFINITY;
        for (int i = tid; i < num_features; i += blockDim.x) {
            local_max = fmaxf(local_max, x[batch_idx * num_features + i]);
        }
        sdata[tid] = local_max;
        __syncthreads();

        // Reduction for max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }
        float max_val = sdata[0];

        // Compute exp and sum
        float local_sum = 0.0f;
        for (int i = tid; i < num_features; i += blockDim.x) {
            float exp_val = __expf(x[batch_idx * num_features + i] - max_val);
            y[batch_idx * num_features + i] = exp_val;
            local_sum += exp_val;
        }
        sdata[tid] = local_sum;
        __syncthreads();

        // Reduction for sum
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float sum_val = sdata[0];

        // Normalize
        for (int i = tid; i < num_features; i += blockDim.x) {
            y[batch_idx * num_features + i] /= sum_val;
        }
    } else {
        // Use warp-based approach for larger feature counts
        int elementsPerWarp = (num_features + num_warps - 1) / num_warps;
        int start = warp_id * elementsPerWarp;
        int end = min(start + elementsPerWarp, num_features);

        float local_max = -INFINITY;
        for (int i = start + lane; i < end; i += WARP_SIZE) {
            local_max = fmaxf(local_max, x[batch_idx * num_features + i]);
        }

        // Warp-level reduction using shuffle
        uint32_t mask = 0xffffffff;
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
        }

        if (lane == 0) sdata[warp_id] = local_max;
        __syncthreads();

        if (tid == 0) {
            float global_max = sdata[0];
            for (int i = 1; i < num_warps; i++) {
                global_max = fmaxf(global_max, sdata[i]);
            }
            sdata[0] = global_max;
        }
        __syncthreads();
        float max_val = sdata[0];

        float local_sum = 0.0f;
        for (int i = start + lane; i < end; i += WARP_SIZE) {
            float exp_val = __expf(x[batch_idx * num_features + i] - max_val);
            y[batch_idx * num_features + i] = exp_val;
            local_sum += exp_val;
        }

        // Warp-level reduction for sum
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        if (lane == 0) sdata[warp_id] = local_sum;
        __syncthreads();

        if (tid == 0) {
            float global_sum = sdata[0];
            for (int i = 1; i < num_warps; i++) {
                global_sum += sdata[i];
            }
            sdata[0] = global_sum;
        }
        __syncthreads();
        float sum_val = sdata[0];

        for (int i = start + lane; i < end; i += WARP_SIZE) {
            y[batch_idx * num_features + i] /= sum_val;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    
    warp_softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in warp_softmax_kernel: %s\n", cudaGetErrorString(err));
    }
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
    m.def("forward", &forward, "Adaptive Softmax forward (CUDA)");
}