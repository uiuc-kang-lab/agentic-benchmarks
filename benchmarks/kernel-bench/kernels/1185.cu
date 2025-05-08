#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses stride loops to have each thread process multiple output elements
// when the total work (N*M*L) exceeds the number of available threads.
// Each output element C[n, m, l] is computed as the dot-product of A[n, m, :] and B[:, l].

template <typename scalar_t>
__global__ void stride_loop_tensor_matmul_kernel(
    const scalar_t* __restrict__ A,  // A: [N, M, K]
    const scalar_t* __restrict__ B,  // B: [K, L]
    scalar_t* __restrict__ C,        // C: [N, M, L]
    int N, int M, int K, int L) {

  int total = N * M * L;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Iterate over the output elements in a strided loop
  for (int i = idx; i < total; i += stride) {
      int n = i / (M * L);
      int rem = i % (M * L);
      int m = rem / L;
      int l = rem % L;

      scalar_t sum = 0;
      // Compute dot-product over the K dimension
      for (int k = 0; k < K; ++k) {
          scalar_t a_val = A[n * M * K + m * K + k];
      scalar_t b_val = B[k * L + l];
      sum += a_val * b_val;
      }
      C[n * M * L + m * L + l] = sum;
  }
}


void stride_loop_tensor_matmul_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {

  int N = A.size(0);
  int M = A.size(1);
  int K = A.size(2);
  int L = B.size(1);
  int total = N * M * L;

  const int threads = 1024;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "stride_loop_tensor_matmul_cuda_forward", ([&] {
      stride_loop_tensor_matmul_kernel<scalar_t><<<blocks, threads>>>(
          A.data_ptr<scalar_t>(),
          B.data_ptr<scalar_t>(),
          C.data_ptr<scalar_t>(),
          N, M, K, L);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error in stride_loop_tensor_matmul_cuda_forward: %s\n", cudaGetErrorString(err));
  }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor stride_loop_tensor_matmul_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);

  auto C = torch::zeros({N, M, L}, A.options());
  stride_loop_tensor_matmul_cuda_forward(A, B, C);
  return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stride_loop_tensor_matmul_forward, "Stride Loop Tensor Matrix Multiplication (CUDA)");
}
