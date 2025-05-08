#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// Kernel that uses grid-stride loops to handle workloads larger than the available threads.
// Each thread block processes one or more (outer, inner) pairs and each thread uses a stride loop
// to cover the "dim" dimension. The reduction is performed in shared memory to compute the argmax.
__global__ void stride_loop_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {
  const int total = outerSize * innerSize;
  // Grid-stride loop over output indices
  for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
    int outer_idx = idx / innerSize;
    int inner_idx = idx % innerSize;
    int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread processes a subset of the `dim` dimension via a stride loop
    float thread_max = -FLT_MAX;
    int thread_arg = 0;
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
      float val = x[start_offset + d * innerSize];
      if (val > thread_max) {
        thread_max = val;
        thread_arg = d;
      }
    }

    // Allocate shared memory for reduction. Partition shared memory into two arrays:
    // one for the max values and one for the corresponding indices.
    extern __shared__ char shared_mem[];
    float* svals = reinterpret_cast<float*>(shared_mem);
    int* sidx = reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

    svals[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_arg;
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        float other = svals[threadIdx.x + s];
        int other_idx = sidx[threadIdx.x + s];
        if (other > svals[threadIdx.x]) {
          svals[threadIdx.x] = other;
          sidx[threadIdx.x] = other_idx;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      indices[idx] = sidx[0];
    }
    __syncthreads();  // Ensure shared memory is ready for next iteration if any
  }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
  TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
  auto x_contig = x.contiguous();
  auto sizes = x_contig.sizes();
  const int ndim = x_contig.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

  int outerSize = 1;
  for (int d = 0; d < dim; d++) {
    outerSize *= sizes[d];
  }
  const int dimSize = sizes[dim];
  int innerSize = 1;
  for (int d = dim + 1; d < ndim; d++) {
    innerSize *= sizes[d];
  }

  // Build output shape by removing the specified dimension
  std::vector<int64_t> out_sizes;
  for (int i = 0; i < ndim; i++) {
    if (i == dim) continue;
    out_sizes.push_back(sizes[i]);
  }
  auto indices = torch::empty(out_sizes, x.options().dtype(torch::kLong));

  // Launch parameters: use a grid-stride loop for the outer/inner dimensions
  const int total = outerSize * innerSize;
  const int threads = 256;
  const int blocks = (total < 1024 ? total : 1024);
  size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));

  stride_loop_argmax_kernel<<<blocks, threads, shared_mem_size>>>(
      x_contig.data_ptr<float>(),
      indices.data_ptr<int64_t>(),
      outerSize,
      dimSize,
      innerSize);

  return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (stride loops for large workloads)");
}
