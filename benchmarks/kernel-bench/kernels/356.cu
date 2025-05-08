#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// First kernel: Partial reduction per row tile
// Each block processes a tile of the K-dimension for a given row and writes a partial sum into an intermediate tensor

template <typename scalar_t>
__global__ void matvec_partial_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> partial,
    int64_t M,
    int64_t K,
    int tile_size) {

    // Determine the row and tile index
    int row = blockIdx.y;
    int tile = blockIdx.x;
    int tid = threadIdx.x;

    // Compute the column range for this tile
    int col_start = tile * tile_size;
    int col_end = col_start + tile_size;
    if (col_end > K) col_end = K;

    scalar_t sum = 0;
    if (row < M) {
        // Loop over the tile in a strided manner
        for (int col = col_start + tid; col < col_end; col += blockDim.x) {
            sum += A[row][col] * B[col];
        }
    }

    // Shared memory reduction
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the block's partial sum to the intermediate output
    if (tid == 0 && row < M) {
        partial[row][tile] = sdata[0];
    }
}

// Second kernel: Final reduction across tiles for each row
// Each block processes one row from the partial results and reduces them to a single value

template <typename scalar_t>
__global__ void matvec_final_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> partial,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int partial_length) {

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Use shared memory for reduction
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);

    scalar_t local_sum = 0;
    // Stride over the partial results for this row
    for (int i = tid; i < partial_length; i += blockDim.x) {
        local_sum += partial[row][i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row][0] = sdata[0];
    }
}

// C++ interface that combines the two-phase reduction
// First, partial sums are computed per tile and stored in an intermediate tensor,
// then a second kernel reduces these partial sums to produce the final matrix-vector product.

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to be 1D
    auto B_flat = B.view({-1});

    // Allocate the final output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Set tile and block sizes
    int threads = 256;
    int tile_size = threads;  // Use one thread per element in a tile
    // Determine how many tiles are needed to cover the K dimension
    int num_tiles = (K + tile_size - 1) / tile_size;

    // Allocate an intermediate tensor to store partial sums (shape: M x num_tiles)
    auto partial = torch::zeros({M, num_tiles}, A.options());

    // Configure grid dimensions for the partial kernel: one block per tile per row
    dim3 grid_partial(num_tiles, M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda_partial", ([&] {
        size_t sharedMemBytes = threads * sizeof(scalar_t);
        matvec_partial_kernel<scalar_t><<<grid_partial, threads, sharedMemBytes>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            partial.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K, tile_size);
    }));

    // Configure grid for the final reduction kernel: one block per row
    // We'll use the same number of threads for reduction; note that num_tiles may be less than 'threads'
    dim3 grid_final(M);
    int final_threads = threads;
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda_final", ([&] {
        size_t sharedMemBytes = final_threads * sizeof(scalar_t);
        matvec_final_kernel<scalar_t><<<grid_final, final_threads, sharedMemBytes>>>(
            partial.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            num_tiles);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with Two-Phase Tiled Reduction (CUDA)");
}
