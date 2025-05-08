#include <torch/extension.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

// Helper function to compute next power of two
__host__ __device__ int next_power_of_two(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

// CUDA kernel for reverse cumulative sum
// Each block processes one independent scan (a "row" in [outer, scan_len, inner] view).
// For small scan lengths (<= 32) a warp-scan using shuffle intrinsics is used,
// avoiding unnecessary __syncthreads(). Otherwise a shared-memory based inclusive scan is applied.

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int scan_len,
                                        const int inner_count,
                                        const int total_scans) {
  // Each block handles one scan
  int scan_idx = blockIdx.x;
  if (scan_idx >= total_scans) return;

  // Compute base offset for this scan assuming tensor shape [outer, scan_len, inner]
  // where total_scans = outer * inner_count and contiguous layout
  int64_t scan_base = (scan_idx / inner_count) * (scan_len * inner_count) + (scan_idx % inner_count);
  int tid = threadIdx.x;
  extern __shared__ scalar_t s_data[];

  // Use different strategies depending on scan length
  if (scan_len <= 32) {
    // Use warp-scan: threads in a warp execute in lockstep so __syncthreads() is not needed.
    unsigned mask = 0xffffffff;
    scalar_t val = 0;
    if (tid < scan_len) {
      // Load element in reversed order: for thread tid, load from position (scan_len - 1 - tid)
      int64_t in_index = scan_base + (scan_len - 1 - tid) * inner_count;
      val = input[in_index];
    }
    // Perform inclusive scan over the warp using shuffle intrinsics
    for (int offset = 1; offset < scan_len; offset *= 2) {
      scalar_t n_val = __shfl_up_sync(mask, val, offset, scan_len);
      if (tid >= offset && tid < scan_len) {
        val += n_val;
      }
    }
    // Reverse the result: each thread writes the value computed by the thread at index (scan_len - 1 - tid)
    if (tid < scan_len) {
      scalar_t result = __shfl_sync(mask, val, scan_len - 1 - tid, scan_len);
      int64_t out_index = scan_base + tid * inner_count;
      output[out_index] = result;
    }
  } else {
    // For longer scans, use shared memory
    extern __shared__ scalar_t s_data[];  // dynamically allocated shared memory
    if (tid < scan_len) {
      int64_t in_index = scan_base + (scan_len - 1 - tid) * inner_count;
      s_data[tid] = input[in_index];
    }
    __syncthreads();

    // Inclusive scan using simple Hillis-Steele algorithm
    for (int offset = 1; offset < scan_len; offset *= 2) {
      if (tid < scan_len && tid >= offset) {
        s_data[tid] += s_data[tid - offset];
      }
      __syncthreads();
    }

    if (tid < scan_len) {
      // Write the flipped result back to output
      int64_t out_index = scan_base + tid * inner_count;
      output[out_index] = s_data[scan_len - 1 - tid];
    }
  }
}

// Host function for fast reverse cumulative sum
// The tensor is interpreted as having shape [outer, scan_len, inner] where:
//   scan_len = x.size(dim)
//   inner    = product(x.size(dim+1), ..., x.size(-1))
//   outer    = x.numel() / (scan_len * inner)

at::Tensor fast_reverse_cumsum(at::Tensor x, int64_t dim) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
  x = x.contiguous();
  auto sizes = x.sizes();
  int ndim = sizes.size();
  TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");
  
  int64_t scan_len = sizes[dim];
  int64_t inner_count = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner_count *= sizes[i];
  }
  int64_t outer = x.numel() / (scan_len * inner_count);
  int64_t total_scans = outer * inner_count;

  auto output = at::empty_like(x);

  // Choose block size: if small scan, use 32 threads (one warp); otherwise, round up to next power of two
  int threads = (scan_len <= 32) ? 32 : next_power_of_two(scan_len);
  // Ensure we do not exceed the maximum threads per block
  threads = (threads > 1024) ? 1024 : threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fast_reverse_cumsum", ([&] {
    const auto* input_ptr = x.data_ptr<scalar_t>();
    auto* output_ptr = output.data_ptr<scalar_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int shared_mem_size = threads * sizeof(scalar_t);
    reverse_cumsum_kernel<scalar_t><<<total_scans, threads, shared_mem_size, stream>>>(input_ptr, output_ptr, scan_len, inner_count, total_scans);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fast_reverse_cumsum, "Fast Reverse cumulative sum along a specified dimension (CUDA)");
}
