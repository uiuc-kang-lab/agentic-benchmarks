#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

__device__ float reduce_max(float val, float* shared_mem, int tid, int stride) {
    shared_mem[tid] = val;
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    return shared_mem[0];
}

__device__ float reduce_sum(float val, float* shared_mem, int tid, int stride) {
    shared_mem[tid] = val;
    __syncthreads();
    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    return shared_mem[0];
}

__global__ void optimized_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    extern __shared__ float shared_mem[];

    // Compute max value in the row
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += stride) {
        max_val = max(max_val, x_row[i]);
    }
    max_val = reduce_max(max_val, shared_mem, tid, stride);

    // Compute exponentials and sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += stride) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }
    sum_val = reduce_sum(sum_val, shared_mem, tid, stride);

    // Normalize the exponentials
    for (int i = tid; i < num_features; i += stride) {
        y_row[i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;

    optimized_softmax_kernel<<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
        return;
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
    m.def("forward", &forward, "Softmax forward (CUDA)");
}