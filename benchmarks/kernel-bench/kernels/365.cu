#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory to tile load the vector B
// Each block processes one row of matrix A and computes the dot product with vector B

template <typename scalar_t>
__global__ void shared_mem_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    // Each block computes one dot product (one row of A)
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Define tile size for loading B into shared memory
    // For simplicity, we use tile size equal to blockDim.x
    const int TILE_SIZE = blockDim.x;

    // Declare shared memory buffer to hold a tile of vector B
    extern __shared__ char shared_mem[];
    scalar_t* sB = reinterpret_cast<scalar_t*>(shared_mem);

    // Each thread will accumulate its partial sum for the dot product
    scalar_t sum = 0;

    // Process the K dimension in tiles
    for (int tile_offset = 0; tile_offset < K; tile_offset += TILE_SIZE) {
        // Determine number of elements in the current tile
        int tile_elems = TILE_SIZE;
        if (tile_offset + TILE_SIZE > K) {
            tile_elems = K - tile_offset;
        }

        // Cooperative loading of the current tile of B into shared memory
        // Each thread loads one or more elements in a strided loop
        for (int j = tid; j < tile_elems; j += block_size) {
            sB[j] = B[tile_offset + j];
        }
        __syncthreads();  // Ensure the tile is fully loaded

        // Each thread processes its portion of the loaded tile
        for (int j = tid; j < tile_elems; j += block_size) {
            // Multiply corresponding element from A and the shared tile of B
            sum += A[row][tile_offset + j] * sB[j];
        }
        __syncthreads();  // Prepare for next tile iteration
    }

    // Perform block-level reduction to sum the partial results from all threads
    // Reuse the shared memory buffer sB for reduction
    sB[tid] = sum;
    __syncthreads();

    // Tree reduction within the block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sB[tid] += sB[tid + stride];
        }
        __syncthreads();
    }

    // The first thread writes the final dot product result to C
    if (tid == 0) {
        C[row][0] = sB[0];
    }
}

// C++ interface that wraps the CUDA kernel

torch::Tensor shared_mem_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to be a 1D tensor if needed
    auto B_flat = B.view({-1});

    // Allocate output tensor C with shape (M, 1)
    auto C = torch::zeros({M, 1}, A.options());

    // Define block and grid sizes: one block per row
    int threads = 256;  // You can adjust this based on performance
    dim3 grid(M);

    // Shared memory size: we need TILE_SIZE elements of type scalar_t
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "shared_mem_matvec_mul_cuda", ([&] {
        size_t sharedMemBytes = threads * sizeof(scalar_t);
        shared_mem_matvec_kernel<scalar_t><<<grid, threads, sharedMemBytes>>>(
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
    m.def("forward", &shared_mem_matvec_mul_cuda, "Matrix-Vector Multiplication using Shared Memory for B (CUDA)");
}
