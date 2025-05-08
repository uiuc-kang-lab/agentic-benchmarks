/*
Combined CUDA kernel for ELU activation using both vectorized load/store (float4) and shared memory tiling.
This kernel processes the bulk of the data in groups of 4 floats for improved memory throughput, while
handling any leftover elements (if the total number of elements is not a multiple of 4) separately.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Process vectorized data using shared memory tiling
// Each thread loads a float4 element into shared memory, computes ELU, and writes back.
__global__ void elu_kernel_vec_shared(const float4* x, float4* out, float alpha, int n4) {
    extern __shared__ float4 tile[]; // Shared memory allocated dynamically
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load vectorized data from global memory into shared memory
    if (globalIdx < n4) {
        tile[tid] = x[globalIdx];
    }
    __syncthreads();

    // Compute the ELU activation on the tile
    if (globalIdx < n4) {
        float4 val = tile[tid];
        float4 result;
        result.x = (val.x > 0.f) ? val.x : alpha * (expf(val.x) - 1.f);
        result.y = (val.y > 0.f) ? val.y : alpha * (expf(val.y) - 1.f);
        result.z = (val.z > 0.f) ? val.z : alpha * (expf(val.z) - 1.f);
        result.w = (val.w > 0.f) ? val.w : alpha * (expf(val.w) - 1.f);
        tile[tid] = result;  // Write result back into shared memory
    }
    __syncthreads();

    // Write results from shared memory back to global memory
    if (globalIdx < n4) {
        out[globalIdx] = tile[tid];
    }
}

// Kernel: Process the tail elements that are not a multiple of 4
__global__ void elu_kernel_tail(const float* x, float* out, float alpha, int offset, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (globalIdx < n) {
        float val = x[globalIdx];
        out[globalIdx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host interface function
// It dispatches two kernel calls: one for the vectorized portion and one for any remaining tail elements.
torch::Tensor elu_cuda_combined(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Determine the number of float4 groups
    int n4 = n / 4;            // number of vectorizable groups
    int remainder = n % 4;       // remaining elements

    const int threads = 256;
    int blocks_vec = (n4 + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float4);

    // If there is at least one vectorized element, process it using the shared memory kernel
    if (n4 > 0) {
        elu_kernel_vec_shared<<<blocks_vec, threads, sharedMemSize>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }

    // Process any remaining tail elements with a scalar kernel
    if (remainder > 0) {
        int tail_offset = n4 * 4;
        int blocks_tail = (remainder + threads - 1) / threads;
        elu_kernel_tail<<<blocks_tail, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            tail_offset,
            n
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_combined, "Combined ELU activation with shared memory and vectorized load (CUDA)");
}
