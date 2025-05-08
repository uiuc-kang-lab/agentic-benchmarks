#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized CUDA kernel using grid-stride loops and loop unrolling
// This kernel computes: output[n,m,l] = sum_{k=0}^{K-1} A[n,m,k]*B[k,l]
// where A is of shape [N, M, K] and B is of shape [K, L].

template <typename scalar_t>
__global__ void grid_unroll_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int total_elements,
    int M, int L, int K) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = idx; index < total_elements; index += stride) {
        // Compute the 3D indices from the flattened index
        int n = index / (M * L);
        int rem = index % (M * L);
        int m = rem / L;
        int l = rem % L;

        scalar_t sum = 0;
        // Precompute offset for A so we avoid recalculating it in the loop
        int offsetA = n * M * K + m * K;

        int k = 0;
        // Unroll the loop by a factor of 4 for improved performance
        #pragma unroll
        for (; k <= K - 4; k += 4) {
            sum += A[offsetA + k    ] * B[(k    ) * L + l];
            sum += A[offsetA + k + 1] * B[(k + 1) * L + l];
            sum += A[offsetA + k + 2] * B[(k + 2) * L + l];
            sum += A[offsetA + k + 3] * B[(k + 3) * L + l];
        }
        // Handle any remaining iterations
        for (; k < K; k++) {
            sum += A[offsetA + k] * B[k * L + l];
        }

        output[index] = sum;
    }
}

// CUDA forward function
void grid_unroll_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    int total_elements = N * M * L;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "grid_unroll_cuda_forward", ([&] {
        grid_unroll_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements, M, L, K);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in grid_unroll_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// Macros for input tensor checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
torch::Tensor grid_unroll_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);

  auto output = torch::zeros({N, M, L}, A.options());
  grid_unroll_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grid_unroll_forward, "Grid-stride loop with unrolling forward (CUDA)");
}
