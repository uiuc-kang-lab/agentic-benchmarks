#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// This kernel distributes each output element's reduction over an entire block,
// ensuring that all threads in the block participate to compute the min over the r-dimension.
// A grid-stride loop is used to assign output indices to blocks, so that workloads are evenly distributed
// across threads and blocks, avoiding underutilization or bottlenecks.

template <typename scalar_t>
__global__ void even_workload_min_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  // total number of output elements (each corresponds to a unique (outer, inner) pair)
  int total_outputs = outer * inner;

  // Process output elements in a grid-stride loop, so that each block can handle multiple outputs
  for (int out_idx = blockIdx.x; out_idx < total_outputs; out_idx += gridDim.x) {
    // Determine the corresponding outer and inner indices
    int outer_idx = out_idx / inner; 
    int inner_idx = out_idx % inner;
    // Compute the base index for the reduction dimension
    int base = outer_idx * (r * inner) + inner_idx;

    // Each thread in the block computes a partial min over the r dimension,
    // processing elements in a strided manner
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
      int idx = base + j * inner;
      scalar_t val = input[idx];
      if (val < local_min) {
        local_min = val;
      }
    }

    // Use shared memory to reduce the partial results within the block
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[threadIdx.x] = local_min;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        scalar_t other = sdata[threadIdx.x + s];
        if (other < sdata[threadIdx.x]) {
          sdata[threadIdx.x] = other;
        }
      }
      __syncthreads();
    }

    // The first thread writes the result for this output element
    if (threadIdx.x == 0) {
      output[out_idx] = sdata[0];
    }
    __syncthreads(); // Ensure all threads have finished before processing the next output
  }
}

// Forward function: prepares tensor dimensions and kernel launch parameters
// Computes the min reduction over the r dimension of an input tensor reshaped as [outer, r, inner]

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: 'outer' is the product of dimensions before 'dim', 'r' is the size of the reduced dimension,
  // and 'inner' is the product of dimensions after 'dim'.
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Build output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Determine kernel launch configuration
  int total_outputs = outer * inner;
  // Choose a grid size that is the minimum of total_outputs and a cap (e.g., 1024 blocks) to ensure even distribution
  int blocks = total_outputs < 1024 ? total_outputs : 1024;
  const int threads = 256;
  int shared_bytes = threads * sizeof(at::ScalarTypeToCPPType<input.scalar_type()>::type);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "even_workload_min_reduce_cuda", ([&] {
    int shared_mem = threads * sizeof(scalar_t);
    even_workload_min_reduce_kernel<scalar_t><<<blocks, threads, shared_mem, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        outer,
        r,
        inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension with even workload distribution (CUDA)");
}
