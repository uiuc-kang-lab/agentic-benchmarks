#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel performing min reduction over r dimension for a chunk of the outer dimension
// Each thread handles one output element from the chunk
template <typename scalar_t>
__global__ void min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (idx >= total) return;
  
  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;
  
  scalar_t min_val = input[base];
  for (int j = 1; j < r; j++) {
    scalar_t curr = input[base + j * inner];
    if (curr < min_val)
      min_val = curr;
  }
  output[idx] = min_val;
}

// Forward function using multi-stream pipelining to overlap kernel execution with memory operations
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: outer = product of dims before 'dim', r = size at 'dim', inner = product of dims after 'dim'
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  auto output = torch::empty(output_shape, input.options());

  // Split work along the outer dimension into chunks for pipelining
  int num_streams = 4; // Use 4 streams for overlapping
  if (outer < num_streams) {
    num_streams = outer;
  }
  int chunk_size = (outer + num_streams - 1) / num_streams;  // number of outer elements per chunk

  // Create CUDA streams
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cuda", ([&] {
    scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    const int threads = 256; // Number of threads per block, should be a multiple of 32 for warp alignment
    
    // Launch a separate kernel for each chunk asynchronously
    for (int s = 0; s < num_streams; s++) {
      int chunk_start = s * chunk_size;
      if (chunk_start >= outer) break;
      int current_chunk = std::min(chunk_size, outer - chunk_start);
      int total_chunk = current_chunk * inner;
      int blocks = (total_chunk + threads - 1) / threads;
      
      // Compute pointer offsets for the chunk
      // For input: each outer element has r*inner elements
      scalar_t* input_chunk = input_ptr + chunk_start * r * inner;
      // For output: each outer element has inner elements
      scalar_t* output_chunk = output_ptr + chunk_start * inner;
      
      min_reduce_kernel<scalar_t><<<blocks, threads, 0, streams[s]>>>(
          input_chunk,
          output_chunk,
          current_chunk,
          r,
          inner);
    }
  }));

  // Synchronize and destroy streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Pipelined min reduction over a specified dimension (CUDA)");
}
