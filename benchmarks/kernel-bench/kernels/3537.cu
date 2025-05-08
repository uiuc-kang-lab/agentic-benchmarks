#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float
__device__ inline float my_exp(float x) {
    return expf(x);
}

__device__ inline void process_element(float x, float& result) {
    result = (x > 0.0f)
        ? x
        : 1.67326324235437728481f * (my_exp(x) - 1.0f);
    result *= 1.05070098735548049342f;
}

__global__ void selu_kernel_shared_memory(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           size_t numel) {
    extern __shared__ float shared_data[];
    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + tid;
    const size_t stride = blockDim.x * gridDim.x;

    // Load data into shared memory
    if (idx < numel) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Process elements in shared memory
    if (idx < numel) {
        float x = shared_data[tid];
        float result;
        process_element(x, result);
        output[idx] = result;
    }

    // Use warp-level primitives for final reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        result += __shfl_down_sync(0xffffffff, result, offset);
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    size_t shared_memory_size = threads * sizeof(float);
    
    selu_kernel_shared_memory<<<blocks, threads, shared_memory_size>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory Optimization (CUDA)");
}