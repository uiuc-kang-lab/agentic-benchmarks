#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that partitions the workload uniformly among threads to avoid per-element conditional branching
// Each thread calculates the number of elements it must process based on a uniform partition of the total work,
// thereby eliminating the per-iteration bound check and reducing warp divergence.

template <typename scalar_t>
__global__ void sigmoid_kernel_no_divergence(const scalar_t* __restrict__ input,
                                                scalar_t* __restrict__ output,
                                                const int64_t size) {
    // Calculate the global thread ID and total number of threads
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t totalThreads = (int64_t) gridDim.x * blockDim.x;

    // Compute how many elements each thread should process
    // Each thread gets 'base' iterations, and the first 'rem' threads get one extra element
    int64_t base = size / totalThreads;
    int64_t rem = size % totalThreads;
    int64_t iters = base + (tid < rem ? 1 : 0);

    // Process the assigned elements without additional branch checks
    // The index of the i-th element for this thread is: tid + i * totalThreads
    // This mapping guarantees that each index is within bounds based on the computed iters
    for (int64_t i = 0; i < iters; i++) {
        int64_t index = tid + i * totalThreads;
        float x = static_cast<float>(input[index]);
        float r = 1.0f / (1.0f + expf(-x));
        output[index] = static_cast<scalar_t>(r);
    }
}


torch::Tensor forward(torch::Tensor input) {
    // Allocate output tensor
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Use a fixed block size of 256 threads
    constexpr int threads = 256;
    // Compute number of blocks. If the total size is smaller than thread count, use 1 block.
    int blocks = (size < threads) ? 1 : ((size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_no_divergence", ([&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        sigmoid_kernel_no_divergence<scalar_t><<<blocks, threads>>>(input_data, output_data, size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) - No Divergence");
}
