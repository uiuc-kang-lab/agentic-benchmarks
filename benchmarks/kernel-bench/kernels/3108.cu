#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 128

template <unsigned warpSize>
__device__ __inline__ float warpReduceMax(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <unsigned warpSize>
__device__ __inline__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    extern __shared__ float sdata[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Max reduction
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x) {
        max_val = max(max_val, x_row[i]);
    }
    
    max_val = warpReduceMax<32>(max_val);
    if (tid % 32 == 0) sdata[tid / 32] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < blockDim.x / 32) ? sdata[tid] : -INFINITY;
        max_val = warpReduceMax<32>(max_val);
        if (tid == 0) sdata[0] = max_val;
    }
    __syncthreads();
    max_val = sdata[0];

    // Exp and sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }

    sum_val = warpReduceSum<32>(sum_val);
    if (tid % 32 == 0) sdata[tid / 32] = sum_val;
    __syncthreads();

    if (tid < 32) {
        sum_val = (tid < blockDim.x / 32) ? sdata[tid] : 0;
        sum_val = warpReduceSum<32>(sum_val);
        if (tid == 0) sdata[0] = sum_val;
    }
    __syncthreads();
    sum_val = sdata[0];

    // Normalize
    for (int i = tid; i < num_features; i += blockDim.x) {
        y_row[i] /= sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    int shared_mem = ((THREADS_PER_BLOCK + 31)/32) * sizeof(float);
    softmax_kernel<<<grid, THREADS_PER_BLOCK, shared_mem>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
