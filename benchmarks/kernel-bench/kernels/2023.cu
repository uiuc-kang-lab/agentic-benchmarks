#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel maps each valid output element (i.e. each (row, col) with row >= col) to a unique thread
// using a 1D grid covering exactly M = N*(N+1)/2 threads. This eliminates wasted threads in the upper
// triangular region and distributes the workload evenly across threads and blocks. Each thread computes
// C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col] with A and B being lower triangular matrices.
// The mapping from a linear index r in [0, M) to (row, col) is derived from the triangular number
// relationship: row = floor((sqrt(8*r + 1) - 1)/2) and col = r - row*(row+1)/2. 
// Although the dot-product loop workload varies with (row, col), the overall work is distributed evenly
// across many threads, avoiding bottlenecks from idle threads in 2D kernels on diagonal blocks.

__global__ void even_workload_triangular_mm_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N,
                                                     int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop over all valid lower triangular elements
    for (int r = idx; r < M; r += stride) {
        // Map linear index r to (row, col) in the lower triangular matrix
        // Using the formula: row = floor((sqrt(8*r + 1) - 1)/2) and col = r - row*(row+1)/2
        float fr = (float)r;
        float tmp = sqrtf(8.0f * fr + 1.0f);
        int row = (int)((tmp - 1.0f) * 0.5f);
        int row_start = (row * (row + 1)) / 2;
        int col = r - row_start;

        // Since the mapping ensures row >= col, we compute the dot product for C[row, col]
        float sum = 0.0f;
        int offsetA = row * N;  // offset for A's row
        // Iterate k from col to row (inclusive) to perform dot product
        for (int k = col; k <= row; k++) {
            sum += __ldg(&A[offsetA + k]) * __ldg(&B[k * N + col]);
        }
        C[offsetA + col] = sum;
    }
}

// PyTorch interface function
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    // Total number of valid output elements in the lower triangle
    int M = N * (N + 1) / 2;
    auto C = torch::empty_like(A);

    // Launch parameters: we launch one thread per lower-triangular element
    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;

    even_workload_triangular_mm_kernel<<<numBlocks, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Even-workload lower triangular matrix multiplication (CUDA)");
}
