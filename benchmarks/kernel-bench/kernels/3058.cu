#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Experimenting with a block size of 512 to maximize occupancy on the H100 GPU
#define BLOCK_SIZE 512
#define WARP_SIZE 32

// CUDA kernel for softmax using block size = 512
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Each thread processes a chunk of elements
    int items_per_thread = (num_features + blockDim.x - 1) / blockDim.x;
    int start = tid * items_per_thread;
    int end = (start + items_per_thread < num_features) ? (start + items_per_thread) : num_features;
    
    // Step 1: Compute local maximum
    float local_max = -INFINITY;
    for (int i = start; i < end; i++) {
        float val = x[batch_idx * num_features + i];
        local_max = fmaxf(local_max, val);
    }
    
    // Warp-level reduction for max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // Shared memory for warp-level max and sum
    extern __shared__ float shared_mem[];
    float* warp_max = shared_mem;              // size: num_warps
    float* warp_sum = shared_mem + num_warps;    // size: num_warps
    
    if (lane == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Thread 0 reduces warp-level maximums to a single block maximum
    if (tid == 0) {
        float global_max = warp_max[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = fmaxf(global_max, warp_max[i]);
        }
        warp_max[0] = global_max;
    }
    __syncthreads();
    float max_val = warp_max[0];
    
    // Step 2: Compute the sum of exponentials
    float local_sum = 0.0f;
    for (int i = start; i < end; i++) {
        float exp_val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = exp_val; // Store temporary exp value
        local_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Thread 0 reduces the warp sums to obtain the global sum
    if (tid == 0) {
        float global_sum = warp_sum[0];
        for (int i = 1; i < num_warps; i++) {
            global_sum += warp_sum[i];
        }
        warp_sum[0] = global_sum;
    }
    __syncthreads();
    float sum_val = warp_sum[0];
    
    // Step 3: Normalize the values
    for (int i = start; i < end; i++) {
        y[batch_idx * num_features + i] /= sum_val;
    }
}

// Host function to launch the CUDA softmax kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    // Shared memory size for warp max and warp sum arrays
    size_t shared_mem_size = 2 * num_warps * sizeof(float);
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
    }
}

// PyTorch binding
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto y = torch::empty_like(x);

    // Launch kernel with block size 512 as optimal configuration
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
