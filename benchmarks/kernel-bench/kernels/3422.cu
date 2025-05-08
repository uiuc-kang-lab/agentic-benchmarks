#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline GELU function for float precision
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Kernel using shared memory and warp-level primitives for reduction
__global__ void gelu_kernel_reduction(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      size_t numel) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < numel; i += stride) {
        sum += gelu_function(input[i]);
    }

    // Store result in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "Only float32 is supported for the optimized reduction version");

    size_t numel = x.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    auto temp_output = torch::empty({blocks}, x.options());
    auto output = torch::empty({1}, x.options());

    // Launch kernel with shared memory size for reduction
    gelu_kernel_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        numel);

    // Final reduction on CPU
    output[0] = temp_output.sum();

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with optimized reduction");
}
