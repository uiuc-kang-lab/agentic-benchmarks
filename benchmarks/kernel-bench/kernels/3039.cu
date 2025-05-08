#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel implementing softmax with stride loops to handle workloads larger than the number of threads
__global__ void stride_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int num_features) {
    // Each block processes one row
    int batch_idx = blockIdx.x;
    int row_start = batch_idx * num_features;
    int tid = threadIdx.x;

    // Use dynamically allocated shared memory for reduction (size = blockDim.x floats)
    extern __shared__ float shared[];

    // Phase 1: Compute the maximum value in the row using stride loop
    float local_max = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = input[row_start + i];
        local_max = fmaxf(local_max, val);
    }
    shared[tid] = local_max;
    __syncthreads();

    // Reduction to obtain row-wise maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    // Phase 2: Compute exponentials and accumulate partial sums using stride loop
    float local_sum = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(input[row_start + i] - row_max);
        output[row_start + i] = exp_val;  // Temporarily store exponentials
        local_sum += exp_val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Reduction to accumulate the sum of exponentials
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    // Phase 3: Normalize the results to obtain softmax probabilities using stride loop
    for (int i = tid; i < num_features; i += blockDim.x) {
        output[row_start + i] /= row_sum;
    }
}

// Host function to launch the stride-based softmax kernel
void softmax_forward_cuda(const float* input, float* output, int batch_size, int num_features) {
    dim3 grid_dim(batch_size);
    dim3 block_dim(THREADS_PER_BLOCK);
    int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    
    stride_softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(input, output, num_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in stride_softmax_kernel: %s\n", cudaGetErrorString(err));
        return;
    }
}

// PyTorch binding forward function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = input.size(0);
    int num_features = input.size(1);

    auto output = torch::empty_like(input);
    softmax_forward_cuda(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, num_features);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride-based Softmax forward (CUDA)");
}
