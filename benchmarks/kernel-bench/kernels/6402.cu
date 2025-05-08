#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Simple reduction kernel: one thread per output element
template <typename scalar_t>
__global__ void sum_reduce_simple_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer_size * inner_size;
  if (idx < total) {
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;
    scalar_t sum = 0;
    // Unroll if possible
    #pragma unroll
    for (int64_t i = 0; i < reduce_size; i++) {
      sum += input[base_idx + i * inner_size];
    }
    output[idx] = sum;
  }
}

// Block-based reduction kernel using shared memory for large reduce_size
// Each block computes one output element by cooperatively summing its reduce dimension
template <typename scalar_t>
__global__ void sum_reduce_block_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
  extern __shared__ char sdata_char[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_char);

  // Each block computes one output element
  int index = blockIdx.x; // index in flattened output (outer_size * inner_size)
  if (index < outer_size * inner_size) {
    int outer_idx = index / inner_size;
    int inner_idx = index % inner_size;
    int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

    scalar_t sum = 0;
    // Each thread processes a chunk of the reduction
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
      sum += input[base_idx + i * inner_size];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      output[index] = sdata[0];
    }
  }
}

// Host function that selects the appropriate kernel based on reduce_size

torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
  // Handle negative dimension
  if (dim < 0) dim += input.dim();

  // Compute dimensions for reduction
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

  // Set output tensor shape: same as input but with dimension 'dim' set to 1
  sizes[dim] = 1;
  auto output = torch::empty(sizes, input.options());

  // Choose kernel based on the size of the reduction dimension
  // If reduce_size is large, use a block-based reduction with shared memory
  constexpr int kReduceThreshold = 64;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
    if (reduce_size > kReduceThreshold) {
      // Each block computes one output element
      int threads = 256; // number of threads per block
      int blocks = outer_size * inner_size; // one block per output element
      size_t shared_mem = threads * sizeof(scalar_t);
      sum_reduce_block_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          reduce_size,
          outer_size,
          inner_size);
    } else {
      // For smaller reduce_size, launch one thread per output element
      int total = outer_size * inner_size;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      sum_reduce_simple_kernel<scalar_t><<<blocks, threads>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          reduce_size,
          outer_size,
          inner_size);
    }
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sum_reduce_cuda, "Sum reduction forward with modular device functions (CUDA)");
}
