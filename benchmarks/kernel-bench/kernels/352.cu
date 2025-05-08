#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum size for vector B that fits in constant memory
#define MAX_B_SIZE 4096

// Declare constant memory for float and double
__constant__ float d_B_float[MAX_B_SIZE];
__constant__ double d_B_double[MAX_B_SIZE];

// Kernel for float using constant memory for B
__global__ void matvec_kernel_const_float(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> A,
          torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> C,
    int64_t M, int64_t K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[row][k] * d_B_float[k];
    }
    C[row][0] = sum;
  }
}

// Kernel for double using constant memory for B
__global__ void matvec_kernel_const_double(
    const torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> A,
          torch::PackedTensorAccessor32<double,2,torch::RestrictPtrTraits> C,
    int64_t M, int64_t K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    double sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[row][k] * d_B_double[k];
    }
    C[row][0] = sum;
  }
}

// C++ interface function that wraps the CUDA kernels
// This function copies the read-only vector B into constant memory
// and launches a kernel that uses it to compute the matrix-vector multiplication

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

  // Ensure the tensors are contiguous
  A = A.contiguous();
  B = B.contiguous();

  int64_t M = A.size(0);
  int64_t K = A.size(1);

  TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
  TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
              "B must be a vector of shape (K,) or (K, 1)");
  
  auto B_flat = B.view({-1});

  // Ensure that K fits in the constant memory buffer
  TORCH_CHECK(K <= MAX_B_SIZE, "Vector B size exceeds constant memory capacity (", MAX_B_SIZE, ")");

  // Allocate output tensor and initialize to zero
  auto C = torch::zeros({M, 1}, A.options());

  int threads = 256;
  int blocks = (M + threads - 1) / threads;

  cudaError_t err;

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      // Copy B to constant memory for float
      err = cudaMemcpyToSymbol(d_B_float, B_flat.data_ptr<float>(), K * sizeof(float));
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for float");
      
      matvec_kernel_const_float<<<blocks, threads>>>(
          A.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          C.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          M, K);
    } else if (std::is_same<scalar_t, double>::value) {
      // Copy B to constant memory for double
      err = cudaMemcpyToSymbol(d_B_double, B_flat.data_ptr<double>(), K * sizeof(double));
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed for double");
      
      matvec_kernel_const_double<<<blocks, threads>>>(
          A.packed_accessor32<double,2,torch::RestrictPtrTraits>(),
          C.packed_accessor32<double,2,torch::RestrictPtrTraits>(),
          M, K);
    }
  }));

  cudaDeviceSynchronize();
  return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA) using constant memory");
}
