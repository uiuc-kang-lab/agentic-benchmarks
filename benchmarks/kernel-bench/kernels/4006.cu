#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Utilize warp shuffles and shared memory to compute ELU efficiently.
__global__ void elu_kernel_warp_shared(const float* x, float* out, float alpha, int n) {
    extern __shared__ float sdata[]; // Shared memory buffer
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    if (globalIdx < n) {
        float val = x[globalIdx];
        val = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
        sdata[tid] = val;
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Reduce within the block using shared memory and warp-level shuffles
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
      out[blockIdx.x] = sdata[0]; // Write block's result to output
    }
}

// Host interface function for vectorized loading/storing and tail handling.
torch::Tensor elu_cuda_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float);

    elu_kernel_warp_shared<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimized, "Warp optimized ELU activation (CUDA)");
}