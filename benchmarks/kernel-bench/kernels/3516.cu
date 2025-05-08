#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double.
template <typename scalar_t>
__device__ __forceinline__ scalar_t my_exp(scalar_t x);

template <>
__device__ __forceinline__ float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ __forceinline__ double my_exp<double>(double x) {
    return exp(x);
}

// CUDA kernel that leverages shared memory to reduce global memory latency.
// Each block loads a tile of input data into shared memory, computes the SELU activation
// and then writes the result back to global memory. Proper synchronization is maintained
// to ensure thread safety.

template <typename scalar_t>
__global__ void selu_kernel_shared(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     size_t numel) {
    extern __shared__ scalar_t s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory into shared memory if within bounds
    if (idx < numel) {
        s_data[threadIdx.x] = input[idx];
    }

    __syncthreads();

    // Process the tile from shared memory and write results back to global memory
    if (idx < numel) {
        scalar_t x = s_data[threadIdx.x];
        scalar_t res = (x > static_cast<scalar_t>(0))
                            ? x
                            : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[idx] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function that launches the SELU kernel with shared memory optimization.
// The shared memory allocated per block is equal to blockDim.x * sizeof(scalar_t).

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(input.scalar_type().id() == c10::ScalarType::Float ? sizeof(float) : sizeof(double));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_shared", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        // Allocate shared memory: one scalar_t per thread
        selu_kernel_shared<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA) using shared memory");
}
