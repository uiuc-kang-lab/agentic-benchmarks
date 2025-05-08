#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the output using an offset and chunk size
template <typename scalar_t>
__global__ void sum_reduce_kernel_stream(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t offset,
    int64_t chunk_size) {

    int global_idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= offset + chunk_size) return;

    int outer_idx = global_idx / inner_size;
    int inner_idx = global_idx % inner_size;

    scalar_t sum = 0;
    int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    
    // Perform reduction along the specified dimension
    for (int i = 0; i < reduce_size; i++) {
        sum += input[base_idx + i * inner_size];
    }
    
    output[global_idx] = sum;
}


// Host function launching the kernel in a pipelined manner using multiple CUDA streams
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Compute sizes for reduction
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor: set the reduced dimension size to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    int64_t total_elements = outer_size * inner_size;

    // Number of streams to use for pipelining
    const int num_streams = 2;
    int64_t chunk_size = (total_elements + num_streams - 1) / num_streams; // ceil division

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        for (int s = 0; s < num_streams; s++) {
            int64_t offset = s * chunk_size;
            int64_t current_chunk = (offset + chunk_size <= total_elements) ? chunk_size : (total_elements - offset);
            if (current_chunk <= 0) break;
            int blocks = (current_chunk + threads - 1) / threads;
            sum_reduce_kernel_stream<scalar_t><<<blocks, threads, 0, streams[s]>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                reduce_size,
                inner_size,
                offset,
                current_chunk);
        }
    }));

    // Synchronize and destroy streams to ensure correctness
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
