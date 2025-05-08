#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Maximum allowed size for the cumulative dimension (must fit in constant memory)
#define MAX_N 1024

// Precomputed reversed indices stored in constant memory for fast repeated read-only access
__constant__ int d_rev_indices[MAX_N];

// This kernel assumes that the reverse cumulative sum is performed along the last dimension.
// Each block handles one "row" (all elements along the cumulative dimension) and uses a parallel inclusive scan
// in shared memory. The kernel loads the row in reversed order using the precomputed indices from constant memory,
// carries out the inclusive scan, and then writes the result back in the original order by reusing the constant memory indices.

__global__ void fast_reverse_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
  // Each block handles one row
  int row = blockIdx.x;
  int tid = threadIdx.x;

  // Allocate shared memory for the current row; its size is specified at kernel launch
  extern __shared__ float sdata[];

  // Load the reversed element into shared memory using constant memory index mapping
  if (tid < N) {
    // d_rev_indices[tid] = N - 1 - tid (precomputed on host)
    sdata[tid] = input[row * N + d_rev_indices[tid]];
  }
  __syncthreads();

  // Perform an in-place inclusive scan (Hillis-Steele algorithm) in shared memory
  // Each iteration adds the value from an offset behind; the use of __syncthreads() ensures correct order
  for (int offset = 1; offset < N; offset *= 2) {
    float tmp = 0.0f;
    if (tid >= offset) {
      tmp = sdata[tid - offset];
    }
    __syncthreads();
    sdata[tid] += tmp;
    __syncthreads();
  }

  // Write the computed cumulative sum back to global memory in the original order.
  // The index mapping is reversed again by using d_rev_indices.
  if (tid < N) {
    output[row * N + d_rev_indices[tid]] = sdata[tid];
  }
}

at::Tensor fast_reverse_cumsum(at::Tensor x, int64_t dim) {
  // Ensure input tensor is contiguous and on CUDA
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
  x = x.contiguous();

  // This fast kernel currently supports cumulative sum only along the last dimension
  TORCH_CHECK(dim == x.dim() - 1, "fast_reverse_cumsum only supports the last dimension");

  // Get the size of the dimension along which we perform the cumulative sum
  int N = x.size(dim);
  TORCH_CHECK(N <= MAX_N, "Dimension size exceeds constant memory limit (", MAX_N, ")");

  // Calculate the number of rows (all dimensions preceding the cumulative dimension flattened)
  int outer_size = x.numel() / N;

  // Allocate output tensor of the same shape
  auto out = at::empty_like(x);

  // Precompute reversed indices on the host; for each index i, reversed index = N - 1 - i
  int h_rev_indices[MAX_N];
  for (int i = 0; i < N; i++) {
    h_rev_indices[i] = N - 1 - i;
  }
  // Copy the precomputed indices to constant memory
  cudaMemcpyToSymbol(d_rev_indices, h_rev_indices, N * sizeof(int));

  // Launch the kernel: one block per row, N threads per block, and allocate shared memory for N floats
  dim3 blocks(outer_size);
  dim3 threads(N);
  size_t shared_mem_size = N * sizeof(float);
  fast_reverse_cumsum_kernel<<<blocks, threads, shared_mem_size>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), N);
  cudaDeviceSynchronize();

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fast_reverse_cumsum, "Fast reverse cumulative sum along the last dimension (CUDA)");
}
