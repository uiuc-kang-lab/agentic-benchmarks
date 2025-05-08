#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA kernel using atomicAdd for inter-warp reduction to avoid extra synchronizations
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t M,
    const int64_t K) {

    // Each block computes one row of the result
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;

    scalar_t sum = 0;
    const scalar_t* row_ptr = A + row * K;
    
    // Each thread processes multiple elements in the row
    for (int64_t k = tid; k < K; k += blockDim.x) {
        sum += __ldg(&row_ptr[k]) * __ldg(&B[k]);
    }

    // Warp-level reduction using shfl_down_sync (no __syncthreads() needed within a warp)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first thread of each warp adds its reduced sum to the output using atomicAdd,
    // thus avoiding an extra __syncthreads() for inter-warp reduction
    if (lane == 0) {
        atomicAdd(&C[row], sum);
    }
}

// C++ function wrapping the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous().view({-1});

    const int64_t M = A_contig.size(0);
    const int64_t K = A_contig.size(1);

    // Allocate output tensor and initialize to zero
    auto C = torch::zeros({M}, A.options());

    dim3 blocks(M);
    dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A_contig.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A_contig.data_ptr<scalar_t>(),
            B_contig.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA) using atomic add reduction");
}
