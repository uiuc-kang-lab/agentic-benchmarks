#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel using shared memory and vectorized operations
__global__ void optimized_elu_kernel(const float4* x, float4* out, float alpha, int n4, int n) {
    extern __shared__ float4 tile[]; // Shared memory for vectorized data
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load vectorized data into shared memory
    if (globalIdx < n4) {
        tile[tid] = x[globalIdx];
    }
    __syncthreads();

    // Compute ELU activation
    if (globalIdx < n4) {
        float4 val = tile[tid];
        float4 result;
        result.x = (val.x > 0.f) ? val.x : alpha * (expf(val.x) - 1.f);
        result.y = (val.y > 0.f) ? val.y : alpha * (expf(val.y) - 1.f);
        result.z = (val.z > 0.f) ? val.z : alpha * (expf(val.z) - 1.f);
        result.w = (val.w > 0.f) ? val.w : alpha * (expf(val.w) - 1.f);
        tile[tid] = result;
    }
    __syncthreads();

    // Write results back to global memory
    if (globalIdx < n4) {
        out[globalIdx] = tile[tid];
    }

    // Handle tail elements
    int tailIdx = globalIdx + n4 * 4;
    if (tailIdx < n) {
        float val = ((float*)x)[tailIdx];
        ((float*)out)[tailIdx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host function
torch::Tensor optimized_elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;

    const int threads = 256;
    int blocks = (n4 + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float4);

    optimized_elu_kernel<<<blocks, threads, sharedMemSize>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        alpha,
        n4,
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_elu_cuda, "Optimized ELU activation (CUDA)");
}