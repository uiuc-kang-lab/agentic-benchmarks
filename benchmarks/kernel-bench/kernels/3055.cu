#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Constant memory for frequently accessed configuration
__constant__ int d_num_features;
__constant__ float d_eps = 1e-6f;

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float shared_mem[];
    float* warp_max = shared_mem;
    float* warp_sum = shared_mem + num_warps;

    // Use constant memory for feature count
    int items_per_warp = (d_num_features + num_warps - 1) / num_warps;
    int warp_start = warp_id * items_per_warp;
    int warp_end = min(warp_start + items_per_warp, d_num_features);

    // First pass: find max value
    float thread_max = -INFINITY;
    #pragma unroll 4
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float val = x[batch_idx * d_num_features + i];
        thread_max = max(thread_max, val);
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Block-level reduction for max
    if (tid == 0) {
        float block_max = warp_max[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();

    float max_val = warp_max[0];

    // Second pass: compute exp and sum
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float val = __expf(x[batch_idx * d_num_features + i] - max_val);
        y[batch_idx * d_num_features + i] = val;
        thread_sum += val;
    }

    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block-level reduction for sum
    if (tid == 0) {
        float block_sum = warp_sum[0];
        #pragma unroll
        for (int i = 1; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum + d_eps; // Add epsilon for numerical stability
    }
    __syncthreads();

    float sum_val = warp_sum[0];

    // Final pass: normalize
    #pragma unroll 4
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        y[batch_idx * d_num_features + i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    // Copy num_features to constant memory
    cudaMemcpyToSymbol(d_num_features, &num_features, sizeof(int));

    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    size_t shared_mem_size = 2 * num_warps * sizeof(float);
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y);
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