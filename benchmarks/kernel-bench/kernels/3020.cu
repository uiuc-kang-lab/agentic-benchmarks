#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256

// CUDA kernel declaration
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features);

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    // Set up grid and block dimensions
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);

    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;

    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    // Optional: check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

// CUDA kernel definition
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block handles one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pointer to the beginning of the current row
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Shared memory for reduction
    extern __shared__ float sdata[];

    // Compute max value in the row
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x_row[i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Reduction to find the max value among threads
    if (tid < stride) {
        sdata[tid] = max_val;
    }
    __syncthreads();

    unsigned int s = stride / 2;
    while (s > 0) {
        if (tid < s && (tid + s) < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
        s >>= 1;
    }

    max_val = sdata[0];
    __syncthreads();

    // Now compute exponentials and sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }

    // Reduction to compute sum of exponentials
    if (tid < stride) {
        sdata[tid] = sum_val;
    }
    __syncthreads();

    s = stride / 2;
    while (s > 0) {
        if (tid < s && (tid + s) < stride) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
        s >>= 1;
    }

    sum_val = sdata[0];
    __syncthreads();

    // Normalize the exponentials
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] = y_row[i] / sum_val;
    }
}

// C++ forward function
torch::Tensor forward(torch::Tensor x) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);

    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);

    return y;
}

// pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}