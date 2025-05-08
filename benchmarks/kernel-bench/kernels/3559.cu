#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// CUDA kernel that applies the SELU activation using shared memory to cache
// input values for each block. Each thread loads its element into shared
// memory, computes the SELU function, then writes the result back to global memory.
// This reduces repeated global memory accesses if the data is frequently reused.

template <typename scalar_t>
__global__ void selu_shared_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    // Allocate shared memory dynamically: one element per thread in the block.
    extern __shared__ scalar_t s_input[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load input from global memory to shared memory, if within bounds
    if (idx < numel) {
        s_data[tid] = input[idx];
    }
    __syncthreads();
    
    // Compute SELU activation if within bounds.
    if (idx < numel) {
        scalar_t x = s_data[tid];
        const scalar_t alpha = static_cast<scalar_t>(1.67326324235437728481);
        const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
        
        // SELU activation: lambda * (x if x > 0 else alpha * (exp(x) - 1))
        scalar_t value = (x > static_cast<scalar_t>(0)) ? x : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[idx] = lambda * value;
    }
}

// Host function that launches the SELU CUDA kernel with shared memory usage.
// Exposed to Python as "forward".

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;  // Tuning block size based on hardware
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_shared", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        // Allocate shared memory: one element per thread
        size_t shared_mem_size = threads * sizeof(scalar_t);
        selu_shared_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory (CUDA)");
}
