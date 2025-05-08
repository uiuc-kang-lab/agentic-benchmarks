#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that computes cumulative product along a given dimension.
// Each thread handles one independent cumulative product chain over the specified dimension.

template <typename scalar_t>
__global__ void cumprod_kernel_stream(
    scalar_t* output,
    const scalar_t* input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_batches) {

    // Loop over independent cumulative product chains using grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_batches;
         idx += blockDim.x * gridDim.x) {

        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        scalar_t product = 1;

        // Sequential cumulative product along the dim axis
        for (int i = 0; i < dim_size; i++) {
            int64_t curr_idx = batch_idx * (dim_size * stride) + i * stride + in_idx;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

// Forward function that overlaps memory transfers with kernel computation using CUDA streams.
// If the input tensor is on CPU, it is asynchronously copied to the GPU,
// the cumulative product is computed on the GPU using a dedicated non-blocking stream,
// and then the output is asynchronously copied back to host pinned memory.

torch::Tensor cumprod_cuda_stream_forward(torch::Tensor input, int64_t dim) {
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_cuda_stream", ([&] {
        cumprod_kernel_stream<scalar_t><<<blocks, threads, 0, kernel_stream>>>(
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
    m.def("forward", &cumprod_cuda_stream_forward, "Cumulative product forward with CUDA streams and overlapping memory transfers");
}
