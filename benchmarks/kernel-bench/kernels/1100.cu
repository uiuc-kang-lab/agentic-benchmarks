#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 128

template <typename scalar_t>
__global__ void warp_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;  // Warp ID within block
    const int lane = tid % WARP_SIZE; // Lane ID within warp
    
    const int row = blockIdx.y * (BLOCK_SIZE/WARP_SIZE) + wid;
    const int col = blockIdx.x * WARP_SIZE + lane;
    
    const int batch = row / M;
    const int m = row % M;

    scalar_t sum = 0;

    if (batch < N && m < M && col < L) {
        for (int k = 0; k < K; k += WARP_SIZE) {
            scalar_t a_val = 0;
            scalar_t b_vals[WARP_SIZE];
            
            if (k + lane < K) {
                a_val = __ldg(&A[batch * M * K + m * K + k + lane]);
            }
            
            if (k + lane < K) {
                b_vals[lane] = __ldg(&B[(k + lane) * L + col]);
            }
            
            #pragma unroll
            for (int offset = 0; offset < WARP_SIZE; ++offset) {
                scalar_t broadcast_a = __shfl_sync(0xffffffff, a_val, offset);
                scalar_t b_val = __shfl_sync(0xffffffff, b_vals[offset], lane);
                
                if (k + offset < K) {
                    sum += broadcast_a * b_val;
                }
            }
        }
        
        output[batch * M * L + m * L + col] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    dim3 block(BLOCK_SIZE);
    dim3 grid((L + WARP_SIZE - 1) / WARP_SIZE,
              (N * M + (BLOCK_SIZE/WARP_SIZE) - 1) / (BLOCK_SIZE/WARP_SIZE));

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "warp_optimized_kernel", ([&] {
        warp_optimized_kernel<scalar_t><<<grid, block>>>(
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

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
    m.def("forward", &module_fn_forward, "Warp optimized tensor-matrix multiplication (CUDA)");
}