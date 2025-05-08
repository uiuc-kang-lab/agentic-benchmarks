/*
Hybrid Softmax Kernel
Combines grid-stride loops (Kernel2) with optimized warp-level reductions using shuffle operations (Kernel1).
Each block processes one row of the input tensor.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Hybrid kernel that uses grid-stride loops and warp-level reductions.
__global__ void hybrid_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    const int blockSize = blockDim.x;  // expected to be THREADS_PER_BLOCK

    // Pointer to the current row
    const float* row_x = x + batch_idx * num_features;
    float* row_y = y + batch_idx * num_features;

    // Phase 1: Compute maximum value in the row using grid-stride loop
    // Initialize with negative infinity
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockSize) {
        thread_max = fmaxf(thread_max, row_x[i]);
    }

    // Intra-warp reduction for max using shuffle
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, offset));
    }

    // Use shared memory to store each warp's maximum
    __shared__ float smax[THREADS_PER_BLOCK / WARP_SIZE];
    int warp_id = tid / WARP_SIZE;
    if ((tid % WARP_SIZE) == 0) {
        smax[warp_id] = thread_max;
    }
    __syncthreads();

    // First thread performs reduction over warp-level maxima
    float max_val;
    if (tid == 0) {
        max_val = smax[0];
        int nWarps = blockSize / WARP_SIZE;
        for (int i = 1; i < nWarps; i++) {
            max_val = fmaxf(max_val, smax[i]);
        }
        smax[0] = max_val;  // store the final max in shared memory slot 0
    }
    __syncthreads();
    max_val = smax[0];

    // Phase 2: Compute exponentials and their sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockSize) {
        float exp_val = __expf(row_x[i] - max_val);
        row_y[i] = exp_val; // store temporarily in output
        thread_sum += exp_val;
    }

    // Intra-warp reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Use shared memory to accumulate warp sums
    __shared__ float ssum[THREADS_PER_BLOCK / WARP_SIZE];
    if ((tid % WARP_SIZE) == 0) {
        ssum[warp_id] = thread_sum;
    }
    __syncthreads();

    // First thread reduces the warp sums
    float sum_val;
    if (tid == 0) {
        sum_val = ssum[0];
        int nWarps = blockSize / WARP_SIZE;
        for (int i = 1; i < nWarps; i++) {
            sum_val += ssum[i];
        }
        ssum[0] = sum_val;  // store final sum
    }
    __syncthreads();
    sum_val = ssum[0];

    // Phase 3: Normalize the exponentials to obtain softmax probabilities
    for (int i = tid; i < num_features; i += blockSize) {
        row_y[i] = row_y[i] / sum_val;
    }
}

// Host function to launch the hybrid softmax kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    hybrid_softmax_kernel<<<grid, block>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in hybrid_softmax_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
}

// PyTorch binding: forward function
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

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Softmax forward (CUDA)");
}
