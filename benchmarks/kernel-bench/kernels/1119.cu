#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    extern __shared__ scalar_t shared_B[];

    int n = blockIdx.x;
    int m = threadIdx.x;
    int l_offset = blockIdx.y * blockDim.x;

    scalar_t sum = 0;
    for (int k = 0; k < K; ++k) {
        // Each thread loads one B value into shared memory
        if(threadIdx.x < L) {
            shared_B[threadIdx.x] = B[k * L + l_offset + threadIdx.x];
        }
        __syncthreads();

        // Perform computation
        for (int l = 0; l < blockDim.x && l_offset + l < L; ++l) {
            scalar_t a_val = A[n * M * K + m * K + k];
            sum += a_val * shared_B[l];
        }
        __syncthreads();
    }
    if(l_offset + m < L) {
        output[n * M * L + m * L + l_offset + m] = sum;
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

    const int threads = 32; // Use a warp size
    const int blocks_x = N;
    const int blocks_y = (L + threads - 1) / threads;

    dim3 blocks(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        // Shared memory size
        int shared_mem_size = threads * sizeof(scalar_t);
        module_fn_cuda_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), N, M, K, L);
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