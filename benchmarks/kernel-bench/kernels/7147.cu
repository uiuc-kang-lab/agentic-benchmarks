#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Structure to hold reduction parameters in constant memory
struct ConstParams {
  int outer;
  int r;
  int inner;
};

// Declare constant memory variable for read-only parameters
__constant__ ConstParams d_params;

// Kernel that performs min reduction over a specified dimension using constant memory
// for frequently accessed parameters. Each block handles one output element corresponding
// to an (outer, inner) pair. The reduction is performed over the r dimension.

template <typename scalar_t>
__global__ void const_mem_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output) {
  // Compute global index for the output element
  int idx = blockIdx.x;
  if (idx >= d_params.outer * d_params.inner) return;

  int outer_idx = idx / d_params.inner;
  int inner_idx = idx % d_params.inner;
  
  // Compute pointer to the start of the reduction segment
  const scalar_t* in_ptr = input + outer_idx * (d_params.r * d_params.inner) + inner_idx;

  // Each thread performs a strided reduction over the r dimension
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < d_params.r; j += blockDim.x) {
    scalar_t val = in_ptr[j * d_params.inner];
    if (val < local_min) {
      local_min = val;
    }
  }

  // Allocate shared memory for reduction
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
  sdata[threadIdx.x] = local_min;
  __syncthreads();

  // Reduce within the block using iterative halving
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) {
      scalar_t other = sdata[threadIdx.x + s];
      if (other < sdata[threadIdx.x]) {
        sdata[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  
  // Unroll the final warp using warp shuffle to minimize divergence
  if (threadIdx.x < 32) {
    scalar_t val = sdata[threadIdx.x];
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      scalar_t other = __shfl_down_sync(mask, val, offset);
      if (other < val) {
        val = other;
      }
    }
    sdata[threadIdx.x] = val;
  }

  // Write the output from the first thread
  if (threadIdx.x == 0) {
    output[idx] = sdata[0];
  }
}

// Host function to set up constant memory and launch the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute dimensions based on the reduction dimension
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create output shape by omitting the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Copy reduction parameters to constant memory on the device
  ConstParams host_params;
  host_params.outer = outer;
  host_params.r = r;
  host_params.inner = inner;
  cudaMemcpyToSymbol(d_params, &host_params, sizeof(ConstParams));

  int total = outer * inner;
  int threads = 256; // Use 256 threads per block
  int blocks = total;  // One block per reduction group

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "const_mem_min_reduce_cuda", ([&] {
    int shared_mem_size = threads * sizeof(scalar_t);
    const_mem_min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size, c10::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction with constant memory on a specified dimension (CUDA)");
}
