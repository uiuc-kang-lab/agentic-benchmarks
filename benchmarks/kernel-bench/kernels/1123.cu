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

    __shared__ scalar_t shared_B[32];  // Fixed size shared memory for a warp

    int n = blockIdx.x;    // Batch dimension
    int m = blockIdx.y;    // M dimension
    int tid = threadIdx.x; // Thread ID within the block
    int l_offset = blockIdx.z * blockDim.x;

    // Each thread handles one element in the L dimension
    if (l_offset + tid >= L) return;

    scalar_t sum = 0;
    
    // Loop over K dimension
    for (int k = 0; k < K; ++k) {
        // Load B into shared memory
        if (l_offset + tid < L) {
            shared_B[tid] = B[k * L + l_offset + tid];
        }
        __syncthreads();

        // Compute partial sum
        scalar_t a_val = A[n * M * K + m * K + k];
        if (l_offset + tid < L) {
            sum += a_val * shared_B[tid];
        }
        
        __syncthreads();
    }

    // Write output
    if (l_offset + tid < L) {
        output[n * M * L + m * L + l_offset + tid] = sum;
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