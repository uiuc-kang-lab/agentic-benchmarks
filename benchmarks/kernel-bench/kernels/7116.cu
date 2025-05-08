#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

// Pipelined reduction kernel: processes a chunk of the outer dimension starting at outer_offset for outer_chunk elements.
// Each thread computes the min over the reduction dimension (r) for one (global_outer, inner_idx) coordinate.

template <typename scalar_t>
__global__ void pipelined_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer_offset,    // starting index in the outer dimension
    const int outer_chunk,     // number of outer indices to process in this chunk
    const int r,               // size of the reduction dimension
    const int inner) {         // product of dimensions after the reduced dimension

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_chunk * inner;
  if (idx >= total) return;

  int local_outer = idx / inner; // index within the current chunk
  int inner_idx = idx % inner;
  int global_outer = outer_offset + local_outer;
  
  // Use shared memory for intermediate results
  extern __shared__ char shared_memory[];
  scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
  
  int tid = threadIdx.x;
  int base = global_outer * (r * inner) + inner_idx;
  scalar_t min_val = input[base];
  
  // Process in chunks to handle large reduction dimensions
  const int chunk_size = 32; // Process warp-size elements at a time
  for (int j = 0; j < r; j += chunk_size) {
    int remaining = min(chunk_size, r - j);
    
    // Load and reduce chunk
    for (int k = 1; k < remaining; k++) {
      int index = global_outer * (r * inner) + (j + k) * inner + inner_idx;
      scalar_t curr = input[index];
      min_val = min(min_val, curr);
    }
  }
  
  // Warp-level reduction using shuffle
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    scalar_t other = __shfl_down_sync(0xffffffff, min_val, offset);
    min_val = min(min_val, other);
  }
  
  // First thread in each warp writes to shared memory
  if (tid % 32 == 0) {
    shared_data[tid / 32] = min_val;
  }
  __syncthreads();
  
  // Final reduction across warps (only if block has multiple warps)
  if (tid < (blockDim.x / 32)) {
    min_val = shared_data[tid];
    #pragma unroll
    for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
      if (i < blockDim.x / 32) {
        min_val = min(min_val, shared_data[i]);
      }
    }
  }
  
  // Write result to the correct position in the output tensor
  if (tid == 0) {
    output[global_outer * inner + inner_idx] = min_val;
}

// The forward function partitions the work along the outer dimension into chunks and assigns each chunk to a CUDA stream.
// This allows overlapping kernel execution with memory operations (e.g. prefetching/pipelined data movement) on the H100 GPU.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: 'outer' is the product of dims before the reduced dimension, 
  // 'r' is the size of the reduced dimension, and 'inner' is the product of dims after the reduced dim.
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build the output shape by removing the reduced dimension.
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim)
      output_shape.push_back(input.size(i));
  }
  auto output = torch::empty(output_shape, input.options());

  // Partition the outer dimension into chunks to enable pipelining using multiple CUDA streams.
  int chunk_size = (outer > 1024) ? 1024 : outer;
  int num_chunks = (outer + chunk_size - 1) / chunk_size;

  // Create a set of non-blocking CUDA streams to overlap kernel execution with memory operations.
  const int num_streams = 2;
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  const int threads = 256;

  // Launch kernels for each chunk on alternate streams.
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    int outer_offset = chunk * chunk_size;
    int outer_chunk = std::min(chunk_size, outer - outer_offset);
    int total_chunk = outer_chunk * inner;
    int blocks = (total_chunk + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "pipelined_min_reduce_cuda", ([&] {
      pipelined_min_reduce_kernel<scalar_t><<<blocks, threads, 0, streams[chunk % num_streams]>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          outer_offset,
          outer_chunk,
          r,
          inner);
    }));
  }

  // Synchronize and destroy the streams to ensure all operations are complete.
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Pipelined min reduction over a specified dimension using CUDA streams");
}
