#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// This kernel performs min reduction over a specified dimension with minimal use of __syncthreads().
// The idea is to synchronize only after writing into shared memory and during the coarse reduction phase.
// Once the number of active threads reaches a warp, warp-synchronous reductions using __shfl_down_sync are used.

template <typename scalar_t>
__global__ void minimal_sync_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

  // Each block processes one output element corresponding to a (outer, inner) pair
  int idx = blockIdx.x;
  if (idx >= outer * inner) return;

  int outer_idx = idx / inner;
  int inner_idx = idx % inner;

  // Compute pointer to the beginning of the reduction segment
  const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

  // Each thread computes a partial minimum over the r dimension via striding
  scalar_t local_min = std::numeric_limits<scalar_t>::max();
  for (int j = threadIdx.x; j < r; j += blockDim.x) {
    scalar_t val = in_ptr[j * inner]; if (val != val) continue; // Skip NaN values
    local_min = (val < local_min) ? val : local_min;
  }

  // Load partial minimum into shared memory
  extern __shared__ char shared[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);
  sdata[threadIdx.x] = local_min;
  __syncthreads(); // Ensure all threads have stored their result

  // Coarse reduction: use __syncthreads() only when threads span multiple warps
  if (blockDim.x > 64) {
    for (unsigned int stride = blockDim.x >> 1; stride >= 64; stride >>= 1) {
      if (threadIdx.x < stride) {
        scalar_t other = sdata[threadIdx.x + stride];
        sdata[threadIdx.x] = (other < sdata[threadIdx.x]) ? other : sdata[threadIdx.x];
      }
      __syncthreads(); // Synchronize after each shared memory update
    }
    // Final warp-level reduction without additional __syncthreads()
    if (threadIdx.x < 64) {
      scalar_t val = sdata[threadIdx.x];
      for (int offset = 32; offset > 0; offset /= 2) {
        scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = (tmp < val) ? tmp : val;
      }
      if (threadIdx.x == 0) {
        output[idx] = val;
      }
    }
  } else {
    // If blockDim.x is 64 or less, all threads are in one warp and no __syncthreads() is needed
    scalar_t val = sdata[threadIdx.x];
    for (int offset = blockDim.x >> 1; offset > 0; offset /= 2) {
      scalar_t tmp = __shfl_down_sync(0xffffffff, val, offset);
      val = (tmp < val) ? tmp : val;
    }
    if (threadIdx.x == 0) {
      output[idx] = val;
    }
  }
}


// Host forward function
// Reshapes input to logical dimensions [outer, r, inner] with reduction along dimension 'dim'
// and launches the minimal_sync_min_reduction_kernel.

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes: outer dimensions (before 'dim'), r (size along 'dim'), inner dimensions (after 'dim')
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Prepare output shape by omitting the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }

  auto output = torch::empty(output_shape, input.options());
  int total = outer * inner;  // Number of reduction groups
  
  // Launch configuration: using a fixed thread count per block; adjust based on r if needed
  int threads = 256;
  int blocks = total;
  
  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "minimal_sync_min_reduce_cuda", ([&] {
    int shared_mem_size = threads * sizeof(scalar_t);
    minimal_sync_min_reduction_kernel<scalar_t><<<blocks, threads, shared_mem_size, 
      c10::cuda::getCurrentCUDAStream().stream()>>>(
          input.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          outer,
          r,
          inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "CUDA kernel min reduction with minimal __syncthreads() usage");
}
