#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized cumulative product kernel with explicit unrolling and CUDA streams

template <typename scalar_t>
__global__ void cumprod_opt_stream_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    // Each thread processes one cumulative product chain
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {
        
        // Compute base index for this chain
        const int batch_idx = idx / stride;
        const int in_idx = idx % stride;
        const int64_t base_idx = batch_idx * (dim_size * stride) + in_idx;
        
        scalar_t product = 1;
        int i = 0;

        // Unrolled loop with factor of 8
        #pragma unroll 8
        for (; i + 7 < dim_size; i += 8) {
            const int64_t idx0 = base_idx + i * stride;
            product *= input[idx0];
            output[idx0] = product;

            const int64_t idx1 = base_idx + (i + 1) * stride;
            product *= input[idx1];
            output[idx1] = product;

            const int64_t idx2 = base_idx + (i + 2) * stride;
            product *= input[idx2];
            output[idx2] = product;

            const int64_t idx3 = base_idx + (i + 3) * stride;
            product *= input[idx3];
            output[idx3] = product;

            const int64_t idx4 = base_idx + (i + 4) * stride;
            product *= input[idx4];
            output[idx4] = product;

            const int64_t idx5 = base_idx + (i + 5) * stride;
            product *= input[idx5];
            output[idx5] = product;

            const int64_t idx6 = base_idx + (i + 6) * stride;
            product *= input[idx6];
            output[idx6] = product;

            const int64_t idx7 = base_idx + (i + 7) * stride;
            product *= input[idx7];
            output[idx7] = product;
        }

        // Process any remaining elements individually
        for (; i < dim_size; i++) {
            const int64_t curr_idx = base_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

// Host forward function that integrates asynchronous memory transfers and kernel computation via streams

torch::Tensor cumprod_optimized(torch::Tensor input, int64_t dim) {
    // If input tensor is on CPU, asynchronously transfer it to GPU
    bool input_on_cpu = !input.is_cuda();
    torch::Tensor input_device = input_on_cpu ? input.to(torch::kCUDA, /*non_blocking=*/true) : input;

    // Allocate output tensor on the GPU
    auto output_device = torch::empty_like(input_device);

    // Create a non-blocking CUDA stream for overlapping memory transfers with computation
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Retrieve tensor properties needed for cumulative product along the given dimension
    auto sizes = input_device.sizes();
    auto strides = input_device.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t total_batches = input_device.numel() / dim_size;

    // Configure kernel launch parameters
    const int threads = 256;
    const int blocks = (total_batches + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_optimized", ([&] {
        cumprod_opt_stream_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            output_device.data_ptr<scalar_t>(),
            input_device.data_ptr<scalar_t>(),
            dim_size,
            stride,
            total_batches
        );
    }));

    torch::Tensor output;
    if (input_on_cpu) {
        // Return output on CPU using asynchronous device-to-host copy with pinned memory
        output = torch::empty_like(output_device, output_device.options().device(torch::kCPU).pinned_memory(true));
        cudaMemcpyAsync(output.data_ptr(),
                        output_device.data_ptr(),
                        output_device.numel() * output_device.element_size(),
                        cudaMemcpyDeviceToHost,
                        stream);
    } else {
        output = output_device;
    }

    // Ensure that all operations in the stream complete before returning
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_optimized, "Optimized cumulative product forward with asynchronous streams and explicit unrolling");
}
