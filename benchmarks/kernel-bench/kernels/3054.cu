#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// This kernel computes the softmax for each row with minimal __syncthreads() usage.
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row (batch index)
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockSize + WARP_SIZE - 1) / WARP_SIZE;

    // Step 1: Compute the maximum value in the row (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockSize) {
        float val = x[batch * num_features + i];
        local_max = fmaxf(local_max, val);
    }
    // Intra-warp reduction using shuffle; no block-level sync needed here
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Write warp-level maximums to shared memory
    __shared__ float s_max[THREADS_PER_BLOCK / WARP_SIZE];
    if (lane == 0) {
        s_max[warp_id] = local_max;
    }
    // Synchronize to ensure all warp max values are in shared memory
    __syncthreads();

    // Let thread 0 reduce the warp-level maximums
    if (tid == 0) {
        float global_max = s_max[0];
        for (int i = 1; i < num_warps; i++) {
            global_max = fmaxf(global_max, s_max[i]);
        }
        s_max[0] = global_max;  // store the global maximum
    }
    // Minimal synchronization: only one barrier to broadcast the global max
    __syncthreads();
    float max_val = s_max[0];

    // Step 2: Compute exponentials and local sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockSize) {
        float exp_val = __expf(x[batch * num_features + i] - max_val);
        y[batch * num_features + i] = exp_val; // store temporary exp values
        local_sum += exp_val;
    }
    // Intra-warp reduction for sum using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Write warp-level sums to shared memory
    __shared__ float s_sum[THREADS_PER_BLOCK / WARP_SIZE];
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
    }
    // Synchronize once to ensure all warp sums are available
    __syncthreads();

    // Let thread 0 compute the global sum
    if (tid == 0) {
        float global_sum = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            global_sum += s_sum[i];
        }
        s_sum[0] = global_sum;
    }
    __syncthreads();
    float sum_val = s_sum[0];

    // Step 3: Normalize the exponentials to produce the softmax output
    for (int i = tid; i < num_features; i += blockSize) {
        y[batch * num_features + i] /= sum_val;
    }
}


// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);

    softmax_kernel<<<grid, block>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ forward function binding for PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

// Pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
