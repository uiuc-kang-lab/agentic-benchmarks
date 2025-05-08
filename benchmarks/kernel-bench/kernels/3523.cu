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

// CUDA kernel that evenly distributes the workload among threads.
// Each thread computes a contiguous segment of the data based on its global thread ID.
// This ensures balanced workload and minimizes underutilization or bottlenecks.

template <typename scalar_t>
__global__ void selu_kernel_even_load_balance(const scalar_t* __restrict__ input,
                                               scalar_t* __restrict__ output,
                                               size_t numel) {
    // Compute a unique thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Calculate the number of elements per thread and the residue
    size_t base = numel / total_threads;
    size_t residue = numel % total_threads;
    
    // Each thread processes base elements, plus one extra if its ID is less than the residue
    size_t start = tid * base + (tid < residue ? tid : residue);
    size_t count = base + (tid < residue ? 1 : 0);
    size_t end = start + count;

    for (size_t i = start; i < end; i++) {
        // Load input using __ldg for potential caching benefits
        scalar_t x = __ldg(&input[i]);
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) *
                                 (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function launching the kernel

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Launch configuration: using 1024 threads per block.
    const int threads = 1024;
    int blocks = (numel + threads - 1) / threads;  // Ensures enough threads are launched

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_even_load_balance_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_even_load_balance<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Even Load Balancing (CUDA)");
}
