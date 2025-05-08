#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel combines tiling across the K dimension with shared memory reduction
// It conditionally avoids atomic operations when only one block per row is launched.

template <typename scalar_t>
__global__ void hybrid_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    // Determine the row index this block is responsible for
    int row = blockIdx.y;
    if (row >= M) return;

    // Each block processes a tile (a segment) of the K dimension
    int tile_offset = blockIdx.x * blockDim.x;
    int tile_end = tile_offset + blockDim.x;
    if (tile_end > K) {
        tile_end = K;
    }

    // Each thread computes a partial sum for its portion of the tile
    scalar_t thread_sum = 0;
    for (int col = tile_offset + threadIdx.x; col < tile_end; col += blockDim.x) {
        thread_sum += A[row][col] * B[col];
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ scalar_t sdata[];
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Perform intra-block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the block's computed partial sum to the output
    // If there's only one block per row, we write directly; otherwise we atomically add
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            C[row][0] = sdata[0];
        } else {
            atomicAdd(&C[row][0], sdata[0]);
        }
    }
}

// C++ interface function wrapping the CUDA kernel

torch::Tensor hybrid_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have as many elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K,1)");
    auto B_flat = B.view({-1});

    // Allocate output tensor and initialize to zero
    auto C = torch::zeros({M, 1}, A.options());

    // Set block size
    int threads = 256;
    // Grid dimensions: tile across K dimension and one row per block in y
    int grid_x = (K + threads - 1) / threads;
    dim3 blocks(grid_x, M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_matvec_cuda", ([&] {
        size_t sharedMemBytes = threads * sizeof(scalar_t);
        hybrid_matvec_kernel<scalar_t><<<blocks, threads, sharedMemBytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

// PyBind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_matvec_cuda, "Hybrid Matrix-Vector Multiplication (CUDA)");
}
