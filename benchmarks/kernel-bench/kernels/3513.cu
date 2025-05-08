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

// CUDA kernel that distributes the workload evenly among threads
// Each thread computes its own start and end indices based on its thread id, ensuring a balanced work distribution.

template <typename scalar_t>
__global__ void selu_kernel_evenly_distributed(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    // Divide the work among threads with remainder handling
    size_t base_work = numel / total_threads;
    size_t residue = numel % total_threads;
    
    // Each thread gets base_work elements, with an extra one if tid < residue
    size_t my_work = base_work + (tid < residue ? 1 : 0);
    size_t start = tid * base_work + (tid < residue ? tid : residue);
    size_t end = start + my_work;

    for (size_t i = start; i < end; i++) {
        // Use __ldg() for read-only load
        scalar_t x = __ldg(&input[i]);
        scalar_t result = (x > static_cast<scalar_t>(0))
                              ? x
                              : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * result;
    }
}

// Host function that launches the evenly distributed workload SELU kernel
// This ensures that each thread in the grid processes a contiguous block of elements, which balances the load
// and avoids idle threads or bottlenecks.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    // Configure kernel launch parameters
    const int threads = 1024;
    // Total threads = blocks * threads, here we choose blocks based on numel to ensure enough threads:
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_evenly_distributed", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_evenly_distributed<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Even Workload Distribution (CUDA)");
}
