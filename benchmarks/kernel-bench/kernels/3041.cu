#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// This kernel uses stride loops to process rows with more features than available threads.
__global__ void stride_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row of the input tensor.
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Pointers to the current row for input and output
    const float* row_x = x + batch_idx * num_features;
    float* row_y = y + batch_idx * num_features;

    // Use shared memory for reduction (max and sum), size equals block size
    extern __shared__ float sdata[];

    // Phase 1: Compute the maximum value in the row using a stride loop
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += block_size) {
        float val = row_x[i];
        local_max = fmaxf(local_max, val);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction to compute max across the block
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Phase 2: Compute the exponentials and their sum using a stride loop
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += block_size) {
        // Compute the exponent
        float exp_val = __expf(row_x[i] - max_val);
        row_y[i] = exp_val; // store temporary exponential value
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction to compute the total sum across the block
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

    // Phase 3: Normalize to get the softmax probabilities
    for (int i = tid; i < num_features; i += block_size) {
        row_y[i] = row_y[i] / sum_val;
    }
}

// Host function to launch the CUDA kernel
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    
    stride_softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in stride_softmax_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ interface using PyTorch tensors
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

// Pybind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA stride loop)");
}
