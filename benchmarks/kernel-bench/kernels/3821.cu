#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

// Compute softplus function in a branchless way
__device__ __forceinline__ float softplus_branchless(float x) {
    float ax = fabsf(x);
    float max_val = (x + ax) * 0.5f;
    return max_val + log1pf(expf(-ax));
}

// Kernel using shared memory for reduction and intra-block synchronization.
__global__ void softplus_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {

    __shared__ float shmem[256]; // Shared memory allocation
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int idx = i; idx < size; idx += stride) {
        float x = input[idx];
        float res = softplus_branchless(x);
        output[idx] = res;
        sum += res;
    }

    shmem[tid] = sum;
    __syncthreads();

    // Reduce within a single block to compute a portion of the total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    // Optionally, accumulate results across blocks (incomplete)
    // if (tid == 0) atomicAdd(&global_sum, shmem[0]);
}

// CUDA entry point
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_shared<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
