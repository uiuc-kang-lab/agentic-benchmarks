#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the product of two lower triangular matrices,
// where for each output element C[i, j] = sum_{k=j}^{i} A[i, k] * B[k, j] if i >= j,
// and C[i, j] = 0 if i < j.  
// Work is evenly distributed by mapping the full 2D output onto a 1D grid-stride loop
// over rows, and each thread loops over columns in that row. This avoids launching
// threads for the entire rectangular matrix and minimizes idle work in upper-triangular
// regions, ensuring balanced workload across threads and blocks.

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    extern __shared__ float sA[];
    // Grid-stride loop over rows to distribute work evenly
    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        // Load the current row of A (only up to index i, since A is lower-triangular) into shared memory
        for (int k = threadIdx.x; k <= i; k += blockDim.x) {
            sA[k] = A[i * N + k];
        }
        __syncthreads();

        // Process row i: each thread handles a subset of columns
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            if (j > i) {
                C[i * N + j] = 0.0f;
            } else {
                float sum = 0.0f;
                // Use shared memory for A values to reduce global memory accesses
                for (int k = j; k <= i; ++k) {
                    sum += sA[k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        __syncthreads();
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
    auto C = torch::empty({N, N}, A.options());

    // Use 256 threads per block. We choose gridDim.x to be min(N, 256) to balance the load
    int blockSize = 256;
    int gridSize = (N < 256) ? N : 256;

    triangular_mm_kernel<<<gridSize, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Lower triangular matrix multiplication with even workload distribution (CUDA)");
}
