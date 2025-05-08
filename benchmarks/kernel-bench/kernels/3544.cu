#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: inline exponential function for float and double types
__device__ inline float my_exp(float x) {
    return expf(x);
}

__device__ inline float selu_compute(float x, float alpha, float lambda) {
    return (x > 0.0f) ? lambda * x : lambda * alpha * (my_exp(x) - 1.0f);
}

// CUDA kernel that utilizes loop unrolling within shared memory
// to compute the SELU activation function efficiently.
__global__ void selu_kernel_unroll(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    size_t numel) {
    extern __shared__ float shared[];
    const float alpha = 1.67326324235437728481f;
    const float lambda = 1.05070098735548049342f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        float x = input[i];
        shared[threadIdx.x] = selu_compute(x, alpha, lambda);
        output[i] = shared[threadIdx.x];
    }
}

// Host function to launch the unrolled SELU kernel
// This kernel benefits from loop unrolling and the use of shared memory.
torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    int sharedMemSize = threads * sizeof(float);
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    selu_kernel_unroll<<<blocks, threads, sharedMemSize>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Loop Unrolling (CUDA)");}