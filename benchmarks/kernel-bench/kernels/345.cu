#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that distributes the dot product work over multiple blocks per row.
// Each block computes a partial dot product over a tile of columns and uses atomicAdd to contribute
// to the final result for that row.

template <typename scalar_t>
__global__ void balanced_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    // Each block processes a tile of columns of size tile_block for a given row.
    const int tile_block = 1024;  // Tiled portion of the K-dimension
    int row = blockIdx.y;  // one block in y-dim corresponds to one row of A

    if (row < M) {
        int col_start = blockIdx.x * tile_block;
        int col_end = (col_start + tile_block < K) ? (col_start + tile_block) : K;
        scalar_t sum = 0;
        
        // Each thread within the block processes a subset of columns in [col_start, col_end)
        for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
            sum += A[row][col] * B[col];
        }
        
        // Use shared memory to reduce the partial sums computed by threads in this block
        extern __shared__ scalar_t sdata[];
        int tid = threadIdx.x;
        sdata[tid] = sum;
        __syncthreads();
        
        // Parallel reduction in the block (assumes blockDim.x is a power of 2)
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        // The first thread in the block adds the block's partial result to the global output
        if (tid == 0) {
            atomicAdd(&C[row][0], sdata[0]);
        }
    }
}

// C++ function wrapping the CUDA kernel

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");

    auto B_flat = B.view({-1});

    // Allocate output tensor and initialize to zero
    auto C = torch::zeros({M, 1}, A.options());

    // Launch configuration: grid.x spans the tiling over the K-dimension, grid.y spans the rows
    const int tile_block = 1024;
    int grid_x = (K + tile_block - 1) / tile_block;
    int grid_y = M;
    dim3 blocks(grid_x, grid_y);
    int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        size_t shared_mem_bytes = threads * sizeof(scalar_t);
        balanced_matvec_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    cudaDeviceSynchronize();

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Balanced Matrix-Vector Multiplication (CUDA)");
}
