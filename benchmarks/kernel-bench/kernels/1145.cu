#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel with stride loop optimization
// This kernel computes: output[n,m,l] = sum_{k=0}^{K-1} A[n,m,k]*B[k,l]
// where A is of shape [N, M, K] and B is of shape [K, L].

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M * L;
    int stride = blockDim.x * gridDim.x;

    for (int index = idx; index < total_elements; index += stride) {
        // Cache index calculations
        int n = index / (M * L);
        int m = (index % (M * L)) / L;
        int l = index % L;
        
        // Pre-calculate base offsets
        int a_base = n * M * K + m * K;
        int b_base = l;
        int out_idx = n * M * L + m * L + l;
        
        scalar_t sum = 0;
        // Main computation loop
        for (int k = 0; k < K; ++k) {
            sum += A[a_base + k] * B[b_base + k * L];
        }
        output[out_idx] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    const int threads = 256;
    const int blocks = (N * M * L + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
      module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
          A.data_ptr<scalar_t>(),
          B.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);

  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}
