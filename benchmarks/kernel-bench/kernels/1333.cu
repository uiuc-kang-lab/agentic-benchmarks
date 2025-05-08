#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel is used when the number of columns (M) is divisible by 4.
// It uses vectorized loads/stores (float4) for improved memory throughput.
// Note: No atomic operations are used, as each thread computes a unique output element.
__global__ void flat_vectorized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t vec_total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);
  
  for (; idx < vec_total; idx += stride) {
    int base_idx = idx * 4;  // Corresponding index in the original array
    int row = base_idx / M;
    float a_val = A[row];

    float4 b_val = B_vec[idx];
    float4 c_val;
    c_val.x = a_val * b_val.x;
    c_val.y = a_val * b_val.y;
    c_val.z = a_val * b_val.z;
    c_val.w = a_val * b_val.w;

    C_vec[idx] = c_val;
  }
}

// This kernel is used when vectorized access is not possible (i.e., M is not divisible by 4).
// Each thread computes a unique output element using a flat grid-stride loop.
// Atomic operations are not needed since there is a one-to-one mapping between threads and output elements.
__global__ void flat_scalar_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total; idx += stride) {
    int row = idx / M;
    C[idx] = A[row] * B[idx];
  }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
  TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
  TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
  TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

  A = A.contiguous();
  B = B.contiguous();

  int64_t N = A.size(0);
  int64_t M = B.size(1);
  int64_t total = N * M;
  auto C = torch::empty({N, M}, B.options());

  int threads = 256;
  
  // If M is divisible by 4, use the vectorized kernel for improved throughput
  if (M % 4 == 0) {
    int64_t vec_total = total / 4;
    int blocks = (vec_total + threads - 1) / threads;
    flat_vectorized_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, vec_total);
  } else {
    int blocks = (total + threads - 1) / threads;
    flat_scalar_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, total);
  }

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Flat diagonal matrix multiplication without unnecessary atomic operations");
}
