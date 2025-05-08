#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// Combined kernel: uses loop unrolling for inner reduction and pipelined streams over outer chunks
template <typename scalar_t>
__global__ void pipelined_unrolled_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer_chunk,  // number of outer elements in this chunk
    const int r,            // reduction dimension length
    const int inner) {      // inner dimension strides

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_chunk * inner;
  if (idx >= total) return;

  // Compute the indices corresponding to the outer and inner dimensions
  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Offset pointer for the current outer element
  // This helps avoid computing (outer_idx * (r*inner) + j*inner + inner_idx) repeatedly
  const scalar_t* row_ptr = input + outer_idx * (r * inner) + inner_idx;
  scalar_t min_val = row_ptr[0];

  // Use loop unrolling for most of the reduction loop
  int unroll_bound = (r / 4) * 4;
  #pragma unroll
  for (int j = 0; j < unroll_bound; j += 4) {
    scalar_t v0 = row_ptr[j * inner];
    scalar_t v1 = row_ptr[(j + 1) * inner];
    scalar_t v2 = row_ptr[(j + 2) * inner];
    scalar_t v3 = row_ptr[(j + 3) * inner];
    min_val = min(min_val, v0);
    min_val = min(min_val, v1);
    min_val = min(min_val, v2);
    min_val = min(min_val, v3);
  }

  // Handle remaining elements if r is not a multiple of 4
  for (int j = unroll_bound; j < r; j++) {
    scalar_t curr = row_ptr[j * inner];
    min_val = min(min_val, curr);
  }

  output[idx] = min_val;
}

// Forward function: sets up pipelining across CUDA streams and launches the optimized kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions: outer = product(dims before dim), r = size at dim, inner = product(dims after dim)
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Construct output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  auto output = torch::empty(output_shape, input.options());

  // Use multiple streams to pipeline kernel launches over outer dimension chunks
  int num_streams = 4;
  if (outer < num_streams) {
    num_streams = outer;
  }
  int chunk_size = (outer + num_streams - 1) / num_streams; // Number of outer elements per chunk

  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "pipelined_unrolled_min_reduce_cuda", ([&] {
    scalar_t* input_ptr = input.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
    const int threads = 256;

    // Launch separate kernels asynchronously on different streams
    for (int s = 0; s < num_streams; s++) {
      int chunk_start = s * chunk_size;
      if (chunk_start >= outer) break;
      int current_chunk = std::min(chunk_size, outer - chunk_start);
      int total_chunk = current_chunk * inner;
      int blocks = (total_chunk + threads - 1) / threads;

      // Adjust pointers for the current chunk
      scalar_t* input_chunk = input_ptr + chunk_start * r * inner;
      scalar_t* output_chunk = output_ptr + chunk_start * inner;

      pipelined_unrolled_min_reduce_kernel<scalar_t><<<blocks, threads, 0, streams[s]>>>(
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
  m.def("forward", &forward, "Pipelined and Unrolled Min Reduction over a specified dimension (CUDA)");
}
