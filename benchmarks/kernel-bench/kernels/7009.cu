#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cuda_fp16.h>

// Structure to hold a value-index pair
template <typename scalar_t>
struct MinPair {
  scalar_t value;
  int index;
};

// Function to combine two MinPair objects and return the one with the smaller value.
// In case of tie, the first is preferred.
template <typename scalar_t>
__device__ __forceinline__ MinPair<scalar_t> min_pair(const MinPair<scalar_t>& a, const MinPair<scalar_t>& b) {
  return (b.index == -1 || a.value <= b.value) ? a : b;
}

// Helper to provide a 'maximum' value for each type for initialization purposes.
// We provide specializations for float, double, int and __half.

template <typename scalar_t>
__device__ __forceinline__ scalar_t max_val();

template <>
__device__ __forceinline__ float max_val<float>() {
  return INFINITY;
}

template <>
__device__ __forceinline__ double max_val<double>() {
  return INFINITY;
}

template <>
__device__ __forceinline__ int max_val<int>() {
  return INT_MAX;
}

template <>
__device__ __forceinline__ __half max_val<__half>() {
  // The largest finite value for half precision is 65504
  return __float2half(65504.0f);
}


// CUDA kernel using shared memory for efficient reduction over the K dimension.
// Each block processes multiple slices via a grid-stride loop. For every slice (i.e. for every combination of outer and inner indices),
// a block of threads cooperatively loads elements along the K dimension into registers and shared memory, and performs a parallel reduction to compute the argmin.

template <typename scalar_t>
__global__ void argmin_shared_kernel(const scalar_t* __restrict__ x,
                                       int64_t* __restrict__ output,
                                       int K,
                                       int64_t outer_size,
                                       int64_t inner_size) {
  // Total number of slices (each slice corresponds to a unique combination of outer and inner indices)
  int64_t total_slices = outer_size * inner_size;

  // Each block processes multiple slices via grid-stride looping
  for (int64_t slice = blockIdx.x; slice < total_slices; slice += gridDim.x) {
    // Determine the outer and inner indices from the slice index
    int64_t outer = slice / inner_size;
    int64_t inner = slice % inner_size;
    
    // Compute pointer to the beginning of the slice.
    // The slice is stored with stride 'inner_size' between consecutive elements along the K dimension.
    const scalar_t* slice_ptr = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Determine the number of threads that will participate in the reduction. Only using min(K, blockDim.x) threads.
    int valid_threads = (K < blockDim.x) ? K : blockDim.x;

    // Each thread computes a local minimum over a subset of the K elements.
    MinPair<scalar_t> local;
    if (threadIdx.x < valid_threads) {
      int k = threadIdx.x;
      local.value = slice_ptr[k * inner_size];
      local.index = k;
      // Stride over K with step = blockDim.x
      for (int j = threadIdx.x + blockDim.x; j < K; j += blockDim.x) {
        scalar_t val = slice_ptr[j * inner_size];
        if (val < local.value) {
          local.value = val;
          local.index = j;
        }
      }
    } else {
      // Threads that do not participate are given a neutral element
      local.value = max_val<scalar_t>();
      local.index = -1;
    }

    // Allocate shared memory for reduction. Dynamically allocated size is blockDim.x * sizeof(MinPair<scalar_t>)
    extern __shared__ char smem[];
    MinPair<scalar_t>* sdata = reinterpret_cast<MinPair<scalar_t>*>(smem);

    sdata[threadIdx.x] = local;
    __syncthreads();

    // Inclusive reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sdata[threadIdx.x] = min_pair(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
      }
      __syncthreads();
    }

    // The first thread writes the result for this slice
    if (threadIdx.x == 0) {
      output[slice] = sdata[0].index;
    }
  }
}


at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

  int dims = x.dim();
  if (dim < 0) dim += dims;
  TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

  // Compute outer_size, K (size along reduction dimension), and inner_size
  int64_t outer_size = 1;
  for (int i = 0; i < dim; i++) {
    outer_size *= x.size(i);
  }
  int K = static_cast<int>(x.size(dim));
  int64_t inner_size = 1;
  for (int i = dim + 1; i < dims; i++) {
    inner_size *= x.size(i);
  }

  // Create output tensor with the reduction dimension removed
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < dims; i++) {
    if (i == dim) continue;
    out_sizes.push_back(x.size(i));
  }
  auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

  // Total number of slices
  int64_t total_slices = outer_size * inner_size;

  // Decide on the kernel launch configuration.
  // Use a fixed number of threads per block (e.g., 256) and a grid size that is at most 1024 blocks, using grid-stride loop for extra slices.
  int threads = 256;
  int blocks = (total_slices < 1024) ? total_slices : 1024;

  // Launch the kernel with dynamic shared memory size of (threads * sizeof(MinPair<scalar_t>)).
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
    const auto* x_data = x.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<int64_t>();
    argmin_shared_kernel<scalar_t><<<blocks, threads, threads * sizeof(MinPair<scalar_t>)>>>(
      x_data,
      output_data,
      K,
      outer_size,
      inner_size
    );
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA) with shared memory");
}
