#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function for single element computation
template <typename scalar_t>
__device__ scalar_t compute_element(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    int n, int m, int l, int K) {
    scalar_t sum = 0;
    for (int k = 0; k < K; ++k) {
        scalar_t a_val = A[n * M * K + m * K + k];
        scalar_t b_val = B[k * l];
        sum += a_val * b_val;
    }
    return sum;
}

// CUDA kernel
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M * L;

    if (idx < total_elements) {
        int n = idx / (M * L);
        int m = (idx % (M * L)) / L;
        int l = idx % L;

        output[n * M * L + m * L + l] = compute_element(A, B, n, m, l, K);
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

    const int threads = 1024;
    const int total_elements = N * M * L;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
      module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
          A.data_ptr<scalar_t>(),
          B.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          N, M, K, L);
    }));

    // Ensure synchronization
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