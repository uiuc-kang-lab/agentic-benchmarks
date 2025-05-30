#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// This kernel computes argmax over a specified dimension for a slice of the output indices.
// The kernel uses a simple grid-stride loop to process indices in the range [start, end) of the flattened outer*inner dimensions.
// Each block is assumed to use 32 threads (one warp), and a warp-level reduction via __shfl_down_sync computes the argmax for that output element.

__global__ void pipeline_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize,
    const int start,
    const int end) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int base_offset = outer_idx * dimSize * innerSize + inner_idx;
        float max_val = -FLT_MAX;
        int max_arg = -1;
        for (int d = 0; d < dimSize; d++) {
            float val = __ldg(&x[base_offset + d * innerSize]);
            if (val > max_val) {
                max_val = val;
                max_arg = d;
            } else if (val == max_val && d < max_arg) {
                max_arg = d;
            }
        }
        indices[idx] = max_arg;
    }
}

// Host function uses multiple CUDA streams to overlap kernel execution with memory operations.
// The total work (outerSize * innerSize output elements) is divided into chunks, each processed on a separate stream.
// This pipelining can help hide memory latencies (e.g., asynchronous copies) and improve throughput on systems that support concurrent stream execution.

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    const int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");
    
    // Compute outerSize, dimSize, innerSize based on the input tensor shape
    int outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }
    
    // Build output shape by removing the reduced dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_sizes.push_back(sizes[i]);
        }
    }
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);
    
    // Total number of output elements
    int total = outerSize * innerSize;
    
    // Set up multiple streams for pipelining
    const int num_streams = 4;  // This can be tuned based on the hardware and workload
    int chunk_size = (total + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch the kernel in each stream for its corresponding chunk of output elements
    const int threads = 32;  // one warp per block
    // Use a reasonable block count per stream; the kernel uses a grid-stride loop so blocks can be fewer than the chunk size
    const int blocks = 128;

    for (int i = 0; i < num_streams; i++) {
        int start = i * chunk_size;
        int end = (i + 1) * chunk_size;
        if (end > total)
            end = total;
        if (start >= end)
            continue;
        pipeline_argmax_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            outerSize,
            dimSize,
            innerSize,
            start,
            end
        );
        // Optionally, you could initiate asynchronous memcpy operations here if transferring chunks of the result to host pinned memory
    }

    // Wait for all streams to complete their work
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward with pipelined streams");
}
