/*
 * This CUDA kernel implements softmax across rows of a 2D tensor,
 * combining warp-level reductions using __shfl_down_sync with a final block reduction
 * in shared memory. This approach minimizes synchronization overhead and shared memory usage,
 * leading to improved efficiency over traditional reduction loops.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel using warp shuffle reductions
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row (batch instance)
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pointers to the current row
    const float* input = x + batch * num_features;
    float* output = y + batch * num_features;

    // Phase 1: Compute the maximum value in the row
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Warp-level reduction to compute max using shuffle intrinsics
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Each warp writes its max to shared memory
    __shared__ float s_max[32]; // Assuming max 32 warps per block
    int warp_id = tid / warpSize;
    if ((tid % warpSize) == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction across warp results by a single thread
    if (tid == 0) {
        int num_warps = (stride + warpSize - 1) / warpSize;
        float max_val = s_max[0];
        for (int i = 1; i < num_warps; i++) {
            max_val = fmaxf(max_val, s_max[i]);
        }
        s_max[0] = max_val;
    }
    __syncthreads();
    float row_max = s_max[0];

    // Phase 2: Compute exponentials and accumulate their sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(input[i] - row_max);
        output[i] = exp_val;   // Store temporary exponential values
        local_sum += exp_val;
    }

    // Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Each warp stores its partial sum in shared memory
    __shared__ float s_sum[32];
    if ((tid % warpSize) == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction for sum by one thread
    if (tid == 0) {
        int num_warps = (stride + warpSize - 1) / warpSize;
        float sum_val = s_sum[0];
        for (int i = 1; i < num_warps; i++) {
            sum_val += s_sum[i];
        }
        s_sum[0] = sum_val;
    }
    __syncthreads();
    float row_sum = s_sum[0];

    // Phase 3: Normalize to produce final softmax probabilities
    for (int i = tid; i < num_features; i += stride) {
        output[i] = output[i] / row_sum;
    }
}

// Host function to launch the CUDA kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    // Launch the kernel; no dynamic shared memory is needed as we use statically allocated shared arrays
    softmax_kernel<<<grid_dim, block_dim>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface (using PyTorch tensors)
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

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA) using warp-level reduction");
}
