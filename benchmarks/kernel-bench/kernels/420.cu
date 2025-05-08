#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using shared memory for intra-block reduction
// and warp-level primitives for final reduction stages.

template <typename scalar_t>
__global__ void matvec_mul_kernel_shared(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    __shared__ scalar_t shared_data[256]; // Fixed size shared memory for up to 256 threads
    const int warp_size = 32;
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row < M) {
        scalar_t sum = 0;

        // Load data into shared memory
        for (int col = lane; col < K; col += warp_size) {
            sum += A[row][col] * B[col];
        }

        // Store partial sum in shared memory
        shared_data[threadIdx.x] = sum;
        __syncthreads();

        // Intra-block reduction using shared memory
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            if (lane < offset) {
                shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
            }
            __syncthreads();
        }

        // Write result from the first thread of each warp
        if (lane == 0) {
            C[row][0] = shared_data[threadIdx.x];
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
    int num_blocks = (M + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_shared<scalar_t><<<num_blocks, threads_per_block, threads_per_block * sizeof(scalar_t)>>>(
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
    m.def("forward", &matvec_mul_cuda, "Optimized Matrix-Vector Multiplication with Shared Memory (CUDA)");
}
