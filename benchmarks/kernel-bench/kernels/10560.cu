#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that computes cumulative product with unrolled loops and streams

template <typename scalar_t>
__global__ void cumprod_unroll8_stream_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {
    // Loop over independent cumulative product chains using grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        scalar_t product = 1;
        int i = 0;

        // Main loop with unroll factor of 8
        #pragma unroll 8
        for (; i + 7 < dim_size; i += 8) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int64_t curr_idx = base_idx + (i + j) * stride;
                product *= input[curr_idx];
                output[curr_idx] = product;
            }
        }

        // Handle remaining elements
        for (; i < dim_size; i++) {
            const int64_t curr_idx = base_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

// Forward function that overlaps memory transfers with kernel computation using CUDA streams.
torch::Tensor cumprod_optimized_stream(torch::Tensor input, int64_t dim) {
    // Check if input is on CPU
    bool input_on_cpu = !input.is_cuda();
    torch::Tensor input_device;
    if (input_on_cpu) {
        // Asynchronously transfer input to GPU
        input_device = input.to(torch::kCUDA, /*non_blocking=*/true);
    } else {
        input_device = input;
    }

    // Allocate output tensor on the same device as input_device
    auto output_device = torch::empty_like(input_device);

    // Create a non-blocking CUDA stream for kernel execution
    cudaStream_t kernel_stream;
    cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking);

    // Retrieve tensor properties to set up the cumulative product
    auto sizes = input_device.sizes();
    auto strides = input_device.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_batches = input_device.numel() / dim_size;

    // Kernel launch parameters
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_optimized_stream", ([&] {
        cumprod_unroll8_stream_kernel<scalar_t><<<blocks, threads, 0, kernel_stream>>>(
            output_device.data_ptr<scalar_t>(),
            input_device.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));

    torch::Tensor output;
    if (input_on_cpu) {
        // Allocate a pinned host tensor for asynchronous device-to-host copy
        output = torch::empty_like(output_device, output_device.options().device(torch::kCPU).pinned_memory(true));
        cudaMemcpyAsync(output.data_ptr(),
                        output_device.data_ptr(),
                        output_device.numel() * output_device.element_size(),
                        cudaMemcpyDeviceToHost,
                        kernel_stream);
    } else {
        output = output_device;
    }

    // Ensure the kernel and memory copies have completed
    cudaStreamSynchronize(kernel_stream);
    cudaStreamDestroy(kernel_stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_optimized_stream, "Optimized cumulative product forward with unrolled loops and CUDA streams");
}