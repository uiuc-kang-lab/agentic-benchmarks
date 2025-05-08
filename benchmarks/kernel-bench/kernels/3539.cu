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

// CUDA kernel that leverages shared memory to cache input data and frequently reused constants
// before applying the SELU activation function. Each block loads a tile of data into shared memory,
// along with two constant values (alpha and lambda) placed in shared memory. Synchronizations
// ensure proper ordering and avoid race conditions.

template <typename scalar_t>
__global__ void selu_kernel_shared(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   size_t numel) {
    // Allocate shared memory: first 2 elements for constants, remaining for data tile
    extern __shared__ char smem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(smem);
    // shared[0]: alpha, shared[1]: lambda
    // Data tile starts at shared + 2
    scalar_t* tile = shared + 2;

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load constants into shared memory once per block
    if (tid == 0) {
        shared[0] = static_cast<scalar_t>(1.67326324235437728481);  // alpha
        shared[1] = static_cast<scalar_t>(1.05070098735548049342);  // lambda
    }
    __syncthreads();

    // Load a tile of input data from global memory into shared memory
    if (global_idx < numel) {
        tile[tid] = input[global_idx];
    }
    __syncthreads();

    // Process the data within shared memory
    if (global_idx < numel) {
        scalar_t x = tile[tid];
        scalar_t res = (x > static_cast<scalar_t>(0))
                          ? x
                          : shared[0] * (my_exp(x) - static_cast<scalar_t>(1));
        res = shared[1] * res;
        tile[tid] = res;
    }
    __syncthreads();

    // Write the processed results back to global memory
    if (global_idx < numel) {
        output[global_idx] = tile[tid];
    }
}

// Host function to launch the shared memory optimized SELU kernel
// The shared memory size is allocated as (blockDim.x + 2) elements to
// accommodate the data tile and the constant values.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_shared_cuda", ([&] {
        int sharedMemSize = (threads + 2) * sizeof(scalar_t);
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_shared<scalar_t><<<blocks, threads, sharedMemSize>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory Optimization (CUDA)");
}
