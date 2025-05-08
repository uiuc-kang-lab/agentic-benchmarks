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

// CUDA kernel that applies the SELU activation using shared memory tiling.
// Each thread loads its corresponding element from global memory into shared memory,
// computes the activation, and writes the result to global memory.
// Note: __syncthreads() is not used here because each thread only accesses its own shared
// memory slot, so no inter-thread synchronization is necessary for correctness.

template <typename scalar_t>
__global__ void selu_kernel_shared_opt(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       size_t numel) {
    // Allocate shared memory dynamically; each thread gets one slot.
    extern __shared__ __align__(sizeof(scalar_t)) char shared_mem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(shared_mem);
    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    const scalar_t alpha  = static_cast<scalar_t>(1.67326324235437728481);
    const scalar_t lambda = static_cast<scalar_t>(1.05070098735548049342);

    // Process elements using a grid-stride loop.
    for (size_t i = global_index; i < numel; i += stride) {
        // Each thread loads its element into shared memory; if out-of-bound, load a dummy value.
        shmem[tid] = (i < numel) ? input[i] : static_cast<scalar_t>(0);
        // No __syncthreads() is needed here since each thread only uses its own data.
        scalar_t x = shmem[tid];
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? x
                              : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = lambda * result;
    }
}

// Host function that launches the CUDA SELU kernel with shared memory optimization.
// It allocates shared memory equal to (threads per block * sizeof(scalar_t)) and avoids
// unnecessary synchronizations by letting each thread work independently on its shared mem slot.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    const int threads = 256;  // number of threads per block
    const int blocks = (numel + threads - 1) / threads;

    // Allocate shared memory: one scalar_t per thread.
    size_t shared_mem_size = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_shared_opt_cuda", ([&] {
        selu_kernel_shared_opt<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory Optimization (CUDA)");
}
