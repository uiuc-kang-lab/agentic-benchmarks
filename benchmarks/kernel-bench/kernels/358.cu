#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void memory_coalesced_hybrid_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {
    
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    __shared__ scalar_t partial_sum[32][32];  // Assuming blockDim.x and blockDim.y are both 32

    scalar_t value = 0;
    if (row < M) {
        for (int tile_col = col; tile_col < K; tile_col += blockDim.x) {
            value += A[row][tile_col] * B[tile_col];
        }
    }
    partial_sum[threadIdx.y][col] = value;
    __syncthreads();

    // Reduce within the block for each row
    if (col == 0 && row < M) {
        scalar_t sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += partial_sum[threadIdx.y][i];
        }
        C[row][0] = sum;
    }
}

torch::Tensor memory_coalesced_hybrid_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    auto B_flat = B.view({-1});

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    
    auto C = torch::zeros({M, 1}, A.options());

    dim3 threads(32, 32); // Use a 32x32 block for better memory coalescing
    int blocks = (M + threads.y - 1) / threads.y;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "memory_coalesced_hybrid_matvec_cuda", ([&] {
        memory_coalesced_hybrid_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A_cont.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &memory_coalesced_hybrid_matvec_cuda, "Memory Coalesced Hybrid Matrix-Vector Multiplication (CUDA)");
}