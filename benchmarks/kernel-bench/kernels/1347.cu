#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Scalar version with manual loop unrolling
__global__ void unrolled_scalar_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t N,
    int64_t M) {
  int row = blockIdx.x;
  if (row < N) {
    float a_val = A[row];
    int offset = row * M;
    // Each thread processes a contiguous chunk of 4 elements
    int start = threadIdx.x * 4;
    int stride = blockDim.x * 4;
    for (int j = start; j < M; j += stride) {
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        int col = j + k;
        if (col < M) {
          C[offset + col] = a_val * B[offset + col];
        }
      }
    }
  }
}

// Vectorized version using float4 with manual loop unrolling
__global__ void unrolled_vectorized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t N,
    int64_t M) {
  int row = blockIdx.x;
  if (row < N) {
    float a_val = A[row];
    int offset = row * M;
    // M is divisible by 4, so reinterpret as float4
    int vecM = M / 4;
    const float4* B_vec = reinterpret_cast<const float4*>(B + offset);
    float4* C_vec = reinterpret_cast<float4*>(C + offset);
    int start = threadIdx.x * 4;
    int stride = blockDim.x * 4;
    for (int j = start; j < vecM; j += stride) {
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        int index = j + k;
        if (index < vecM) {
          float4 b_val = B_vec[index];
          float4 c_val;
          c_val.x = a_val * b_val.x;
          c_val.y = a_val * b_val.y;
          c_val.z = a_val * b_val.z;
          c_val.w = a_val * b_val.w;
          C_vec[index] = c_val;
        }
      }
    }
  }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
  TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
  TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
  TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: the size of A must match the first dimension of B");

  A = A.contiguous();
  B = B.contiguous();

  int64_t N = A.size(0);
  int64_t M = B.size(1);
  auto C = torch::empty({N, M}, B.options());

  int threads = 256;
  // One block per row
  dim3 grid(N);
  if (M % 4 == 0) {
    unrolled_vectorized_diag_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);
  } else {
    unrolled_scalar_diag_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);
  }
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Diagonal matrix multiplication with manual loop unrolling");
}
