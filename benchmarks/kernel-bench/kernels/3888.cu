#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel using atomic operations where necessary
__global__ void softsign_kernel_atomic(const float* x, float* out, int num_elements) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Use shared memory to coalesce global memory loads
    if (idx < num_elements) {
        shared_data[tid] = x[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    if (idx < num_elements) {
        float val = shared_data[tid];
        float result = val / (1.0f + fabsf(val));

        // Use atomic operations only where necessary
        atomicExch(&out[idx], result);
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    // Configure launch parameters
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    size_t shared_memory_size = threads * sizeof(float);
    softsign_kernel_atomic<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fast Softsign activation with atomic operation optimization (CUDA)");
}