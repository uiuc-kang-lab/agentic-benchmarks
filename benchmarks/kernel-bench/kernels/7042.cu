#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Kernel: Each block computes the min reduction along the specified dimension for one output element
// by loading parts of the reduction dimension into shared memory and reducing them cooperatively.

template <typename scalar_t>
__global__ void min_reduce_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

  // Compute the output element index that this block is responsible for
  int idx = blockIdx.x; // idx in [0, outer*inner)
  int outer_idx = idx / inner;
  int inner_idx = idx % inner;
  int base = outer_idx * (r * inner) + inner_idx;

  // Shared memory allocation
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  int tid = threadIdx.x;
  // Initialize with the maximum possible value as the neutral element
  scalar_t my_min = std::numeric_limits<scalar_t>::max();

  // Each thread processes multiple elements along the reduction dimension
  for (int j = tid; j < r; j += blockDim.x) {
    int pos = base + j * inner;
    scalar_t val = input[pos];
    my_min = (val < my_min) ? val : my_min;
  }

  // Store each thread's partial result in shared memory
  sdata[tid] = my_min;
  __syncthreads();

  // Perform tree reduction in shared memory, synchronizing only when necessary
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      scalar_t other = sdata[tid + s];
      sdata[tid] = (other < sdata[tid]) ? other : sdata[tid];
    }
    __syncthreads();
  }

  // Use warp-level reduction without additional __syncthreads()
  if (tid < 32) {
    volatile scalar_t* vsdata = sdata;
    vsdata[tid] = (vsdata[tid + 32] < vsdata[tid]) ? vsdata[tid + 32] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 16] < vsdata[tid]) ? vsdata[tid + 16] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 8] < vsdata[tid]) ? vsdata[tid + 8] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 4] < vsdata[tid]) ? vsdata[tid + 4] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 2] < vsdata[tid]) ? vsdata[tid + 2] : vsdata[tid];
    vsdata[tid] = (vsdata[tid + 1] < vsdata[tid]) ? vsdata[tid + 1] : vsdata[tid];
  }
  
  // Write the final result from thread 0
  if (tid == 0)
    output[idx] = sdata[0];
}


// Host function to set up the kernel launch
torch::Tensor forward(torch::Tensor input, int64_t dim) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Compute sizes for outer, reduction dimension (r), and inner
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Construct the output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Launch one block per output element
  int blocks = outer * inner;
  int threads = (r < 256 ? r : 256);

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_shared_cuda", ([&] {
    min_reduce_shared_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t),
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
  m.def("forward", &forward, "Min reduction over a specified dimension using shared memory (CUDA)");
}
