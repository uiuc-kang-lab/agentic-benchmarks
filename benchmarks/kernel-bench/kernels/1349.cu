#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized kernel using grid-stride loops for matrices where M is divisible by 4
__global__ void stride_loop_vectorized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t vec_total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // reinterpret B and C as float4 pointers
  const float4* B_vec = reinterpret_cast<const float4*>(B);
  float4* C_vec = reinterpret_cast<float4*>(C);

  // Process elements in grid-stride loop
  for (int i = idx; i < vec_total; i += stride) {
    // Compute the corresponding base index in the scalar array
    int base_idx = i * 4;
    // Boundary check (should always pass if total is divisible by 4)
    if (base_idx < N * M) {
      // Determine row by dividing the base index by number of columns
      int row = base_idx / M;
      float a_val = A[row];
      
      // Load 4 consecutive floats from B
      float4 b_val = B_vec[i];
      float4 c_val;
      c_val.x = a_val * b_val.x;
      c_val.y = a_val * b_val.y;
      c_val.z = a_val * b_val.z;
      c_val.w = a_val * b_val.w;
      
      C_vec[i] = c_val;
    }
  }
}

// Scalar kernel using grid-stride loops
__global__ void stride_loop_scalar_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t total,
    const int64_t M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // Each thread processes multiple elements in a grid-stride loop
  for (int i = idx; i < total; i += stride) {
    int row = i / M;
    C[i] = A[row] * B[i];
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
  
  // Use vectorized kernel if M is divisible by 4
  if (M % 4 == 0) {
    int64_t vec_total = total / 4;  // total number of float4 elements
    int blocks = (vec_total + threads - 1) / threads;
    stride_loop_vectorized_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, vec_total);
  } else {
    int blocks = (total + threads - 1) / threads;
    stride_loop_scalar_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), total, M);
  }
  
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Stride loop diagonal matrix multiplication");
}
