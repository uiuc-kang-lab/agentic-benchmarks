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

// CUDA kernel that evenly partitions the workload among all threads.
// Each thread computes its starting index and the number of elements it should process,
// ensuring an even division of work and minimal underutilization or bottlenecks.

template <typename scalar_t>
__global__ void selu_kernel_even_partitioned(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    // Calculate unique thread id in the entire grid
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // Compute the base number of elements per thread and the extra elements to distribute
    const size_t base_work = numel / total_threads;
    const size_t extra = numel % total_threads;

    // Each thread gets an extra element if its tid is less than 'extra'
    const size_t work = base_work + (tid < extra ? 1 : 0);
    const size_t start = tid * base_work + (tid < extra ? tid : extra);
    
    for (size_t i = 0; i < work; i++) {
        size_t idx = start + i;
        if (idx < numel) {
            // Load input using read-only cache
            scalar_t x = __ldg(&input[idx]);
            // Compute SELU activation: if (x > 0) then x else alpha*(exp(x)-1)
            scalar_t result = (x > static_cast<scalar_t>(0))
                                  ? x
                                  : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
            output[idx] = static_cast<scalar_t>(1.05070098735548049342) * result;
        }
    }
}

// Host function that launches the SELU activation kernel using evenly partitioned workload distribution.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Configure kernel launch parameters
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;  // Ensure enough threads are launched

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_even_partitioned_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_even_partitioned<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Even Partitioned Workload (CUDA)");
}
