#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define block size for reduction. Using 128 threads per block.
#define BLOCK_SIZE 128
#define WARP_SIZE 32

// Kernel where each block computes one output element of the lower triangular matrix product.
// It computes: C[row, col] = sum_{k=col}^{row} A[row*N + k] * B[k*N + col]
// The index (row, col) is obtained by mapping the block index 'm' (0 <= m < total_elements) via m = row*(row+1)/2 + col.
// The reduction is performed in two stages: first, a shared-memory reduction across the block; then a final warp-level reduction using volatile memory accesses.
__global__ void shared_reduce_triangular_mm_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N,
                                                    int total_elements) {
    // Each block is assigned to one lower triangular element
    int m = blockIdx.x;  
    if (m >= total_elements) return;

    // Invert the mapping: m = row*(row+1)/2 + col, with 0 <= col <= row < N
    float m_f = static_cast<float>(m);
    int row = static_cast<int>((sqrtf(8.0f * m_f + 1.0f) - 1.0f) * 0.5f);
    int triangular_start = row * (row + 1) / 2;  // number of elements before current row
    int col = m - triangular_start;

    // Compute the dot product for the (row, col) element in the lower-triangular matrix
    // Only indices k from col to row contribute.
    float sum = 0.0f;
    for (int k = col + threadIdx.x; k <= row; k += blockDim.x) {
        sum += A[row * N + k] * B[k * N + col];
    }

    // Allocate shared memory for intra-block reduction
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    // Reduce in steps: first, reduce across the block until only 32 elements remain
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction using volatile shared memory to avoid bank conflicts
    if (threadIdx.x < 32) {
        volatile float* smem = sdata;
        smem[threadIdx.x] += smem[threadIdx.x + 32];
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    // The first thread in the block writes the result
    if (threadIdx.x == 0) {
        C[row * N + col] = sdata[0];
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Total number of lower triangular elements (including the diagonal)
    int total_elements = N * (N + 1) / 2;

    // Launch one block per output element. Each block uses BLOCK_SIZE threads.
    dim3 grid(total_elements);
    dim3 block(BLOCK_SIZE);

    shared_reduce_triangular_mm_kernel<<<grid, block>>>(
         A.data_ptr<float>(),
         B.data_ptr<float>(),
         C.data_ptr<float>(),
         N,
         total_elements
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Reduction Triangular Matrix Multiplication (CUDA)");
}
