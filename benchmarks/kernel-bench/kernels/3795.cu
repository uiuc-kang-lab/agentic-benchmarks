#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use tensors with pinned memory and multiple streams to overlap memory transfer and computation
// Utilize CUDA streams for overlapping memory copy with kernel execution for better throughput

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = __ldg(&input[idx]);
        
        // Softplus computation
        if (x > 20.0) {
            output[idx] = x;
        } else if (x < -20.0) {
            output[idx] = exp(x);
        } else {
            if (x > 0) {
                output[idx] = x + log1p(exp(-x));
            } else {
                output[idx] = log1p(exp(x));
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    // Enable pinned memory for fast access
    auto options = input.options().pinned_memory(true);
    auto output = torch::empty_like(input, options);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Create CUDA stream for better overlap
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        // Copy data to device using asynchronous copy
        cudaMemcpyAsync(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                        size * sizeof(scalar_t), cudaMemcpyHostToDevice, stream);

        // Launch kernel
        softplus_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);

        // Copy result back to host
        cudaMemcpyAsync(output.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        size * sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
    }));

    // Synchronize stream to ensure all operations are completed
    cudaStreamSynchronize(stream);

    // Destroy the stream after use
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
