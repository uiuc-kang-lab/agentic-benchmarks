#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

// Device function for warp-level reduction
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function for block-level reduction
__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARPS_PER_BLOCK];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < WARPS_PER_BLOCK) ? shared[lane] : 0;
    if (warp_id == 0) val = warp_reduce_sum(val);

    return val;
}

// Kernel for matrix-vector multiplication
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t M,
    const int64_t K)
{
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;

    scalar_t sum = 0;
    const scalar_t* row_ptr = A + row * K;

    #pragma unroll 4
    for (int64_t k = tid; k < K; k += BLOCK_SIZE) {
        sum += row_ptr[k] * B[k];
    }

    sum = block_reduce_sum(sum);

    if (tid == 0) {
        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    auto B_flat = B_contig.view({-1});
    auto C = torch::zeros({M}, A.options());
    
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(M);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A_contig.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));
    
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}