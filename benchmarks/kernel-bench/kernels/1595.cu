#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

// This kernel uses a 1D thread mapping to cover only the valid upper-triangular indices
// by flattening the triangular region into a single dimension. Each thread computes its
// corresponding (row, col) pair using a closed-form inversion of the cumulative counts.
__global__ void upper_triangular_matmul_kernel_1d(const float* A, const float* B, float* C, int N, int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        // Compute row and col from tid using closed-form formulas.
        // The total number of elements before row r is: T(r) = r * N - (r * (r - 1)) / 2.
        // Solve for row: tid = T(row) + offset, where offset = col - row and 0 <= offset < (N - row).
        // Using quadratic inversion: row = floor((2*N + 1 - sqrt((2*N + 1)^2 - 8 * tid)) / 2).
        float fN = (float)N;
        float temp = sqrtf((2.0f * fN + 1.0f) * (2.0f * fN + 1.0f) - 8.0f * (float)tid);
        int row = (int)((2.0f * fN + 1.0f - temp) / 2.0f);
        int rowStart = row * N - (row * (row - 1)) / 2;  // starting tid for this row
        int col = row + (tid - rowStart);

        // Compute dot product only over the valid range [row, col]
        float sum = 0.0f;
        for (int k = row; k <= col; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// The host function sets up a 1D kernel launch that covers exactly the upper-triangular elements
// thereby avoiding wasted threads compared to a 2D grid where half the threads do no work.
// The result remains correct as we compute the sum for each valid (row, col) position.

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    int numElements = (N * (N + 1)) / 2;  // total elements in the upper triangle

    int threads = 256;
    int blocks = (numElements + threads - 1) / threads;

    upper_triangular_matmul_kernel_1d<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, numElements
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication with 1D triangular indexing");
}
