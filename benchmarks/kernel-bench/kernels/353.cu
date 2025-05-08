#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for vector B
__constant__ float B_const[1024];  // Adjust size as needed, ensure it fits within constant memory limits

// CUDA kernel for matrix-vector multiplication using constant memory for vector B
// Each block processes a tile of columns for a single row and atomically accumulates its partial result

template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    // Determine the row this block is processing
    int row = blockIdx.y;
    
    // Each block processes a tile of the K-dimension
    int tile_offset = blockIdx.x * blockDim.x;
    int tile_end = tile_offset + blockDim.x;
    if (tile_end > K) {
        tile_end = K;
    }

    scalar_t local_sum = 0;
    // Ensure the row is valid
    if (row < M) {
        // Each thread in the block processes elements within the tile in a strided manner
        for (int col = tile_offset + threadIdx.x; col < tile_end; col += blockDim.x) {
            local_sum += A[row][col] * B_const[col];
        }
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Perform reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Use a single atomic add per block to accumulate the block's partial sum to the final result
    if (threadIdx.x == 0 && row < M) {
        atomicAdd(&(C[row][0]), sdata[0]);
    }
}

// C++ interface function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are CUDA tensors
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");
    auto B_flat = B.view({-1});

    // Copy B to constant memory
    cudaMemcpyToSymbol(B_const, B_flat.data_ptr<scalar_t>(), K * sizeof(scalar_t));

    // Allocate output tensor and initialize to zero
    auto C = torch::zeros({M, 1}, A.options());

    // Configure block and grid dimensions
    int threads = 256;
    // Each block processes a tile of 'threads' elements in the K dimension
    int grid_x = (K + threads - 1) / threads;
    dim3 blocks(grid_x, M);

    // Launch the kernel with shared memory allocated per block
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        size_t sharedMemBytes = threads * sizeof(scalar_t);
        matvec_mul_kernel<scalar_t><<<blocks, threads, sharedMemBytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with Constant Memory (CUDA)");
}
