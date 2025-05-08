#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256

// CUDA kernel declaration
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features);

// CUDA forward function
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((num_features + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, batch_size);

    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;

    softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
    }
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    extern __shared__ float sdata[];

    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        float val = x[batch_idx * num_features + i];
        max_val = max(max_val, val);
    }

    if (threadIdx.x < blockDim.x) {
        sdata[threadIdx.x] = max_val;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    max_val = sdata[0];
    __syncthreads();

    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = exp_val;
        sum_val += exp_val;
    }

    if (threadIdx.x < blockDim.x) {
        sdata[threadIdx.x] = sum_val;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    sum_val = sdata[0];
    __syncthreads();

    for (int i = tid; i < num_features; i += stride) {
        y[batch_idx * num_features + i] /= sum_val;
    }
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");}