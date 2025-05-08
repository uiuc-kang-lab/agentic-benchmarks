#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// The base kernel: computes mean reduction over a dimension
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t outer_size,
    int64_t dim_size, 
    int64_t inner_size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;
    
    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;
    const int input_offset = outer_idx * dim_size * inner_size + inner_idx;
    
    scalar_t sum = 0;
    for (int i = 0; i < dim_size; i++) {
        sum += input[input_offset + i * inner_size];
    }
    
    output[tid] = sum / dim_size;
}

// Host function: overlaps memory transfers with computation using CUDA streams
// If the input tensor is on CPU, it partitions the input along the outer dimension,
// and uses asynchronous memory copies and kernel launches on multiple streams
// to overlap host-to-device transfers, computation, and device-to-host transfers.

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute sizes
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Create output tensor with the reduced dimension removed
    auto output_sizes = sizes;
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // If input is already on CUDA, launch the kernel directly on the default stream
    if (input.is_cuda()) {
        const int threads = 256;
        const int blocks = (outer_size * inner_size + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
            mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                dim_size,
                inner_size
            );
        }));
        return output;
    }

    // For CPU input, we use asynchronous memory transfers to overlap computation
    // Ensure the input is contiguous
    input = input.contiguous();

    // Decide on number of streams for pipelining (e.g., use 4 or less if outer_size is small)
    int num_streams = outer_size >= 4 ? 4 : 1;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Partition the outer dimension among the streams
    int base_chunk = outer_size / num_streams;
    int remainder = outer_size % num_streams;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        // For each stream, process a chunk
        for (int i = 0; i < num_streams; i++) {
            int chunk_outer = base_chunk + (i < remainder ? 1 : 0);
            if (chunk_outer == 0) continue;
            
            // Compute the starting index along the outer dimension for this chunk
            int start_outer = i * base_chunk + (i < remainder ? i : remainder);
            
            // Calculate number of elements for this chunk
            size_t chunk_input_elems = static_cast<size_t>(chunk_outer) * dim_size * inner_size;
            size_t chunk_output_elems = static_cast<size_t>(chunk_outer) * inner_size;
            size_t chunk_input_bytes = chunk_input_elems * sizeof(scalar_t);
            size_t chunk_output_bytes = chunk_output_elems * sizeof(scalar_t);
            
            // Allocate device memory for input and output chunks for this stream
            scalar_t* d_input_chunk;
            scalar_t* d_output_chunk;
            cudaMalloc(&d_input_chunk, chunk_input_bytes);
            cudaMalloc(&d_output_chunk, chunk_output_bytes);
            
            // Compute host pointers for the current chunk
            scalar_t* h_input_ptr = input.data_ptr<scalar_t>() + start_outer * dim_size * inner_size;
            scalar_t* h_output_ptr = output.data_ptr<scalar_t>() + start_outer * inner_size;
            
            // Asynchronously copy input chunk from host to device
            cudaMemcpyAsync(d_input_chunk, h_input_ptr, chunk_input_bytes, cudaMemcpyHostToDevice, streams[i]);
            
            // Launch the kernel for this chunk
            int num_elements = chunk_outer * inner_size;
            int threads = 256;
            int blocks = (num_elements + threads - 1) / threads;
            mean_reduce_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                d_input_chunk,
                d_output_chunk,
                chunk_outer,
                dim_size,
                inner_size
            );
            
            // Asynchronously copy the result from device to host
            cudaMemcpyAsync(h_output_ptr, d_output_chunk, chunk_output_bytes, cudaMemcpyDeviceToHost, streams[i]);
            
            // Free device memory for this chunk
            cudaFree(d_input_chunk);
            cudaFree(d_output_chunk);
        }
    }));

    // Synchronize all streams to ensure completion
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with overlapped memory transfers (CUDA)");
}
