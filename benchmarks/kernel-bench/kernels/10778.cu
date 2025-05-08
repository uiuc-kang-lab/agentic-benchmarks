#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

// CUDA kernel implementing reverse cumsum on each contiguous row (scan_dim) using a shared-memory parallel scan
// The algorithm reverses the input row, performs an inclusive scan (Hillis-Steele) in shared memory for coalesced access,
// and then writes the result back in the proper order so that output[i] = sum_{j=i}^{n-1} input[j].

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        int scan_dim) {
  // Each block processes one row (contiguous segment of length scan_dim).
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int offset = row * scan_dim;

  // Allocate two buffers in shared memory (using dynamic shared memory).
  extern __shared__ char smem[];
  // Partition shared memory into two arrays for ping-pong buffering
  scalar_t* s_in = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_out = s_in + scan_dim;

  // Load reversed input into shared memory ensuring coalesced access:
  if (tid < scan_dim) {
    s_in[tid] = input[offset + (scan_dim - 1 - tid)];
  }
  // For threads with tid >= scan_dim, do nothing (they remain unused)
  __syncthreads();

  // Perform an inclusive scan (Hillis-Steele algorithm) in shared memory
  // The scan is performed over 'scan_dim' elements.
  for (int d = 1; d < scan_dim; d *= 2) {
    scalar_t add = (tid >= d && tid < scan_dim) ? s_in[tid - d] : static_cast<scalar_t>(0);
    __syncthreads();
    if (tid < scan_dim) {
      s_out[tid] = s_in[tid] + add;
    }
    __syncthreads();
    if (tid < scan_dim) {
      s_in[tid] = s_out[tid];
    }
    __syncthreads();
  }

  // Write the scanned result back in reverse order to obtain the final reverse cumulative sum
  if (tid < scan_dim) {
    output[offset + (scan_dim - 1 - tid)] = s_in[tid];
  }
}

// Host function to launch the reverse cumsum kernel
at::Tensor reverse_cumsum_coalesced(at::Tensor x, int64_t dim) {
  // Ensure the tensor is contiguous and on CUDA
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
  x = x.contiguous();

  // If the scan dimension is not the innermost (stride 1), permute so that it is.
  bool needs_permute = false;
  std::vector<int64_t> perm, inv_perm;
  int dims = x.dim();
  if (dims > 1 && dim != dims - 1) {
    needs_permute = true;
    perm.resize(dims);
    inv_perm.resize(dims);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim], perm[dims - 1]);
    // Compute inverse permutation
    for (int i = 0; i < dims; i++) {
      inv_perm[perm[i]] = i;
    }
    x = x.permute(perm).contiguous();
  }

  // After permutation, the dimension to scan is the last dimension
  int64_t scan_dim = x.size(-1);
  int64_t outer_dim = x.numel() / scan_dim;  // number of rows to process
  auto output = torch::empty_like(x);

  // Determine block size as the next power of 2 >= scan_dim to ensure all elements are processed
  int block_size = 1;
  while (block_size < scan_dim) {
    block_size *= 2;
  }

  dim3 grid(outer_dim);
  dim3 block(block_size);
  // Allocate shared memory: two arrays of size 'block_size' each
  size_t shared_mem_size = 2 * block_size * sizeof(x.scalar_type());

  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "reverse_cumsum_coalesced", ([&] {
    reverse_cumsum_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        static_cast<int>(scan_dim));
  }));

  // If a permutation was applied, invert it to restore original dimensions
  if (needs_permute) {
    // Compute inverse permutation tensor
    std::vector<int64_t> inv_perm_long(inv_perm.begin(), inv_perm.end());
    output = output.permute(inv_perm_long).contiguous();
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reverse_cumsum_coalesced, "Reverse cumulative sum with memory coalescing (CUDA)");
}
