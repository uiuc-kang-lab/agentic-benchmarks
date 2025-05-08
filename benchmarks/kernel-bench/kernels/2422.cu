#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs using shared memory reduction and warp-level primitives
// Each block computes one element of the output matrix C

template <typename scalar_t>
__global__ void matmul_transpose_shared_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
  // Compute global output element index from blockIdx.x
  int idx = blockIdx.x;
  int row = idx / N;
  int col = idx % N;
  if (row >= M || col >= N) return;

  int tid = threadIdx.x;
  scalar_t sum = 0;
  // Each thread computes a portion of the dot product over K
  for (int k = tid; k < K; k += blockDim.x) {
    // A is stored as transposed: element A[k, row] -> A[k * M + row]
    // B is stored as transposed: element B[col, k] -> B[col * K + k]
    sum += A[k * M + row] * B[col * K + k];
  }

  // Allocate shared memory for reduction
  extern __shared__ char shared_mem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
  sdata[tid] = sum;
  __syncthreads();

  // Intra-block reduction in shared memory (tree-based reduction)
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Final warp-level reduction using __shfl_down_sync
  if (tid < 32) {
    // Load the value from shared memory for this thread
    scalar_t val = sdata[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    sdata[tid] = val;
  }

  // Write the final result to C
  if (tid == 0) {
    C[row * N + col] = sdata[0];
  }
}

// CUDA interface that launches the optimized kernel

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
  // Get dimensions: A is (K x M) and B is (N x K) due to transposition
  const int K = A.size(0);
  const int M = A.size(1);
  const int N = B.size(0);

  // Create output tensor C of dimensions (M x N)
  auto C = torch::empty({M, N}, A.options());

  // Launch configuration: one block per output element
  const int threads = 256;
  const int blocks = M * N;
  
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_shared_kernel", ([&] {
    const int shared_size = threads * sizeof(scalar_t);
    matmul_transpose_shared_kernel<scalar_t><<<blocks, threads, shared_size>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K);
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using shared memory reduction (CUDA)");
}
