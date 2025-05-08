#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using 2D thread block mapping for matrix-vector multiplication
// Each block processes multiple rows (blockDim.y) and each thread in x-dimension reduces over the K dimension

template <typename scalar_t>
__global__ void matvec_2d_kernel(const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 scalar_t* __restrict__ C,
                                 const int M,
                                 const int K) {
    // Map 2D block: each block processes blockDim.y rows
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    scalar_t sum = 0;

    // Each thread computes a partial dot-product over the row's K-dimension
    if (row < M) {
        for (int k = tx; k < K; k += blockDim.x) {
            sum += A[row * K + k] * B[k];
        }
    }

    // Allocate shared memory for partial sums; layout: [blockDim.y][blockDim.x]
    extern __shared__ scalar_t sdata[]; // Correctly declare shared memory
    int tid = threadIdx.y * blockDim.x + tx;
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction along the x-dimension (within each row) in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride && row < M) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in each row writes the final result to output
    if (tx == 0 && row < M) {
        C[row] = sdata[threadIdx.y * blockDim.x];
    }
}

// Host function wrapper
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    A = A.contiguous();
    B = B.contiguous();

    const int M = A.size(0);
    const int K = A.size(1);

    // Flatten B in case it is not already 1D
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M}, A.options());

    // Define 2D block dimensions: x-dim for reduction across K, y-dim for rows per block
    const int BLOCK_THREADS = 256;
    const int BLOCK_ROWS = 4;  // Each block processes 4 rows
    dim3 threads(BLOCK_THREADS, BLOCK_ROWS);
    dim3 blocks((M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        const size_t sharedMemSize = BLOCK_THREADS * BLOCK_ROWS * sizeof(scalar_t);
        matvec_2d_kernel<scalar_t><<<blocks, threads, sharedMemSize>>>(
            A.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with 2D Indexing (CUDA)");
}
