#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel evenly distributes the workload among all threads by computing
// a unique start and end index for each thread based on the total number of threads.
// This ensures that no thread is overburdened while others remain idle,
// which helps eliminate bottlenecks and improve overall kernel performance.

template <typename scalar_t>
__global__ void balanced_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    // Total number of threads in the grid
    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Evenly divide the workload among threads
    int quotient = size / totalThreads;
    int remainder = size % totalThreads;
    int start, count;

    // Distribute the remainder among the first 'remainder' threads
    if (tid < remainder) {
        count = quotient + 1;
        start = tid * count;
    } else {
        count = quotient;
        start = remainder * (quotient + 1) + (tid - remainder) * quotient;
    }
    int end = start + count;

    // Process the assigned segment for this thread
    for (int i = start; i < end; i++) {
        scalar_t val = input[i];
        output[i] = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Use a fixed number of threads per block
    const int threads = 256;
    // Ensure that the total number of threads covers all elements
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "balanced_relu_kernel", ([&] {
        balanced_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced Load ReLU CUDA kernel");
}
