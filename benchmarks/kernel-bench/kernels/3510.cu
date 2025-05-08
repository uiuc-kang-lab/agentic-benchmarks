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

// CUDA kernel that distributes the load evenly by computing a contiguous work range per thread.
// Each thread calculates its start and end indices to process a balanced portion of the input.
// This ensures that threads and blocks are fully utilized without bottlenecks.

template <typename scalar_t>
__global__ void selu_kernel_balanced(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      size_t numel) {
    size_t total_threads = blockDim.x * gridDim.x;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the number of elements each thread should process (ceiling division)
    size_t work_per_thread = (numel + total_threads - 1) / total_threads;
    size_t start = tid * work_per_thread;
    size_t end = start + work_per_thread;
    if (end > numel) end = numel;

    for (size_t i = start; i < end; i++) {
        scalar_t x = input[i];
        scalar_t y = (x > static_cast<scalar_t>(0))
                         ? x
                         : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * y;
    }
}

// Host function that launches the balanced CUDA SELU kernel.
// The workload is evenly distributed across threads and blocks to avoid underutilization.

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Configure the kernel launch parameters for high occupancy.
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;  

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_balanced_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_balanced<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA) with Balanced Workload Distribution");
}
