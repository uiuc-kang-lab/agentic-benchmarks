/*
Combined CUDA kernel for cumulative product.
This kernel uses a two-pass approach: each block processes one cumulative product chain by splitting the work among threads.
Each thread computes the product of its assigned segment using loop unrolling for inner-loop optimization.
Then, an exclusive scan (computed via shared memory) calculates the offset for each thread so that the cumulative product is correctly computed across the chain.
The host-side forward function leverages asynchronous memory transfer and CUDA streams to overlap transfer and computation.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Combined kernel: two-phase cumulative product with loop unrolling in each thread's segment

template <typename scalar_t>
__global__ void cumprod_combined_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    // Each block processes one independent cumulative product chain
    int chain_id = blockIdx.x;
    if (chain_id >= total_chains) return;

    // Decode chain index: determine batch index and in-dimension index
    int batch_idx = chain_id / stride;
    int in_idx = chain_id % stride;
    const int64_t base = batch_idx * (dim_size * stride) + in_idx;

    // Each thread in the block will process a contiguous segment of the chain
    const int t = threadIdx.x;
    const int T = blockDim.x;
    // Divide the chain evenly among threads
    int chunk = (dim_size + T - 1) / T;
    int start = t * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // FIRST PASS: Each thread computes the product of its assigned segment using loop unrolling
    scalar_t local_prod = 1;
    int i = start;
    #pragma unroll 4
    for (; i + 3 < end; i += 4) {
        int64_t idx1 = base + (i) * stride;
        int64_t idx2 = base + (i + 1) * stride;
        int64_t idx3 = base + (i + 2) * stride;
        int64_t idx4 = base + (i + 3) * stride;
        local_prod *= input[idx1] * input[idx2] * input[idx3] * input[idx4];
    }
    for (; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }

    // Use shared memory to collect each thread's local product
    extern __shared__ scalar_t s_local[];
    s_local[t] = local_prod;
    __syncthreads();

    // Compute the exclusive prefix product (offset) for this thread
    // Offset is product of all s_local elements from threads with id < t
    scalar_t offset = 1;
    for (int j = 0; j < t; j++) {
        offset *= s_local[j];
    }
    __syncthreads();

    // SECOND PASS: Each thread recomputes the cumulative product on its segment using its offset
    scalar_t prod = offset;
    i = start;
    #pragma unroll 4
    for (; i + 3 < end; i += 4) {
        int64_t idx1 = base + i * stride;
        prod *= input[idx1];
        output[idx1] = prod;
        
        int64_t idx2 = base + (i + 1) * stride;
        prod *= input[idx2];
        output[idx2] = prod;
        
        int64_t idx3 = base + (i + 2) * stride;
        prod *= input[idx3];
        output[idx3] = prod;
        
        int64_t idx4 = base + (i + 3) * stride;
        prod *= input[idx4];
        output[idx4] = prod;
    }
    for (; i < end; i++) {
        int64_t idx = base + i * stride;
        prod *= input[idx];
        output[idx] = prod;
    }
}


// Forward function that prepares data, launches the kernel with asynchronous stream, and handles CPU/GPU data transfers

torch::Tensor cumprod_combined_forward(torch::Tensor input, int64_t dim) {
    // If input is on CPU, asynchronously transfer it to GPU
    bool input_on_cpu = !input.is_cuda();
    torch::Tensor input_device;
    if (input_on_cpu) {
        input_device = input.to(torch::kCUDA, /*non_blocking=*/true);
    } else {
        input_device = input;
    }

    // Allocate output tensor on the same device as input
    auto output_device = torch::empty_like(input_device);

    // Create a non-blocking CUDA stream to overlap memory transfers with kernel execution
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Retrieve tensor properties to set up cumulative product
    auto sizes = input_device.sizes();
    auto strides = input_device.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    // Total number of independent cumulative product chains
    int64_t total_chains = input_device.numel() / dim_size;

    // Choose 256 threads per block. Each block processes one chain.
    int threads = 256;
    dim3 blocks(total_chains);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_device.scalar_type(), "cumprod_combined_forward", ([&] {
        cumprod_combined_kernel<scalar_t><<<blocks, threads_per_block, threads * sizeof(scalar_t), stream>>>(
            output_device.data_ptr<scalar_t>(),
            input_device.data_ptr<scalar_t>(),
            dim_size,
            stride_val,
            total_chains
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
                        stream);
    } else {
        output = output_device;
    }

    // Ensure kernel execution and memory copies are complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_combined_forward, "Combined cumulative product forward with loop unrolling, two-phase scan, and CUDA streams");
}
