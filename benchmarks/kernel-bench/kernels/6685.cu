#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tunable chunk size for partitioning the reduction dimension
#define CHUNK_SIZE 1024

// Kernel to perform partial product reduction over a chunk of the reduction dimension
// Each block computes one output element over the given chunk.
// Parameters:
//   input: pointer to the input tensor data
//   partial_results: pointer to the partial results for this chunk (pre-offset by the host code)
//   dim_size: full size of the reduction dimension
//   stride: stride for the reduction dimension
//   chunk_offset: starting index in the reduction dimension for this chunk
//   chunk_size: number of elements in this chunk
__global__ void partial_prod_kernel(const float* input, float* partial_results, int dim_size, int stride, int chunk_offset, int chunk_size) {
    int out_idx = blockIdx.x;  // Each block handles one output element
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float prod = 1.0f;
    int end = chunk_offset + chunk_size;
    if (end > dim_size) {
        end = dim_size;
    }
    
    // Each thread in the block computes a partial product over the chunk in a strided manner
    for (int j = chunk_offset + tid; j < end; j += blockDim.x) {
        prod *= input[out_idx + j * stride];
    }
    
    sdata[tid] = prod;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the partial product result for this output element
    if (tid == 0) {
        partial_results[out_idx] = sdata[0];
    }
}

// Kernel to combine partial results from all chunks for each output element
// Each block handles one output element by multiplying across the partial products computed
// for that element from different chunks.
__global__ void final_prod_kernel(const float* partial_results, float* output, int num_chunks, int num_elements) {
    int out_idx = blockIdx.x;  // One block per output element
    // Use only one thread per block for this final accumulation
    if (threadIdx.x == 0) {
        float prod = 1.0f;
        // Multiply the partial results from each chunk
        for (int i = 0; i < num_chunks; i++) {
            prod *= partial_results[i * num_elements + out_idx];
        }
        output[out_idx] = prod;
    }
}

// Forward function: performs product reduction over a specified dimension
// by partitioning the reduction dimension into chunks and launching asynchronous 
// kernels (each on its own CUDA stream) to compute partial reductions. Finally, a
// final kernel combines these partial results to produce the final output.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Prepare output shape by removing the reduced dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());
    
    int num_elements = output.numel();
    int stride = x.stride(dim);
    
    // Partition the reduction dimension into chunks
    int num_chunks = (dim_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Allocate an intermediate tensor for partial results from each chunk
    // The shape is [num_chunks, num_elements] in row-major order
    auto partial_sizes = std::vector<int64_t>{num_chunks, num_elements};
    torch::Tensor partial = torch::empty(partial_sizes, x.options());

    // Kernel launch configuration for partial reduction kernels
    int threads = 256;
    int blocks = num_elements;  // One block per output element
    size_t shared_mem_size = threads * sizeof(float);

    // Create non-blocking CUDA streams for overlapping execution
    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    const float* input_ptr = x.data_ptr<float>();
    float* partial_ptr = partial.data_ptr<float>();

    // Launch a partial reduction kernel for each chunk on its own stream
    for (int i = 0; i < num_chunks; i++) {
        int chunk_offset = i * CHUNK_SIZE;
        int current_chunk_size = CHUNK_SIZE;
        if (chunk_offset + current_chunk_size > dim_size) {
            current_chunk_size = dim_size - chunk_offset;
        }
        // Compute pointer offset for the current chunk in the partial tensor
        float* chunk_partial_ptr = partial_ptr + i * num_elements;
        partial_prod_kernel<<<blocks, threads, shared_mem_size, streams[i]>>>(
            input_ptr,
            chunk_partial_ptr,
            dim_size,
            stride,
            chunk_offset,
            current_chunk_size
        );
    }

    // Synchronize all streams to ensure partial kernels have completed
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Launch final kernel to combine partial results from all chunks
    final_prod_kernel<<<blocks, 32>>>(partial_ptr, output.data_ptr<float>(), num_chunks, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension with stream pipelining (CUDA)");
}
