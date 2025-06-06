#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel combining coalesced memory access and grid-stride loop for better occupancy
// Each warp computes one output element (one row dot product) using coalesced memory access.
// No divergent branching within warps.

template <typename scalar_t>
__global__ void matvec_mul_kernel_uniform(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    const int warp_size = 32;
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int row = blockIdx.x * warps_per_block + warp_id;

    // Ensure work only with valid rows
    if (row < M) {
        scalar_t sum = 0;
        // Each thread in warp processes multiple elements with stride, utilizing shared memory for coalesced access.
        #pragma unroll 4
        for (int64_t col = lane; col < K; col += warp_size) {
            sum += A[row][col] * B[col];
        }

        // Use a fixed pattern for warp-level reduction
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Use 256 threads per block (8 warps) for better occupancy
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    
    // Calculate optimal number of blocks based on SM count
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int blocks_per_sm = 2; // Allow multiple blocks per SM for better occupancy
    int m_val = static_cast<int>(M);
    int num_blocks = min(
        (m_val + warps_per_block - 1) / warps_per_block,
        num_sms * blocks_per_sm
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_uniform<scalar_t><<<num_blocks, threads_per_block>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Uniform Warp Reduction Matrix-Vector Multiplication (CUDA)");
}