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

// CUDA kernel that applies the SELU activation using shared memory tiles.
// __syncthreads() is used only when necessary to ensure shared memory consistency.

template <typename scalar_t>
__global__ void selu_kernel_shared(const scalar_t* __restrict__ input,
                                     scalar_t* __restrict__ output,
                                     size_t numel) {
    extern __shared__ char shared_mem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int tid = threadIdx.x;
    // grid-stride loop: each iteration processes a tile of blockDim.x elements
    const int blockStride = blockDim.x * gridDim.x;
    for (size_t base = blockIdx.x * blockDim.x; base < numel; base += blockStride) {
        const size_t idx = base + tid;
        // Load one element from global memory into shared memory if within bounds
        if (idx < numel) {
            tile[tid] = input[idx];
        }
        // Synchronize to ensure the entire tile is loaded
        __syncthreads();

        // Compute SELU activation using the value in shared memory
        if (idx < numel) {
            const scalar_t x = tile[tid];
            const scalar_t alpha  = static_cast<scalar_t>(1.67326324235437728481);
            const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);
            const scalar_t res = (x > static_cast<scalar_t>(0)) 
                                   ? x 
                                   : alpha * (my_exp(x) - static_cast<scalar_t>(1));
            output[idx] = lambda * res;
        }
        // Synchronize before reusing shared memory in the next iteration
        __syncthreads();
    }
}

// Host function that launches the CUDA SELU kernel using shared memory.
// The shared memory size is set to blockDim.x * sizeof(scalar_t).

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    const int threads = 512;
    // Each block processes a tile of 'threads' elements per iteration
    const int blocks = (numel + threads - 1) / threads;
    const size_t shared_bytes = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_shared_cuda", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        scalar_t *output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_shared<scalar_t><<<blocks, threads, shared_bytes>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory (CUDA)");
}
