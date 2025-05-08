#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

// This kernel maps each matrix row to the x-dimension of the grid and dynamically assigns
// the valid column indices using the y-dimension of the grid plus threadIdx.x. 
// For a given row, the valid column indices start at 'row'. Thus, the calculation:
//    col = row + (blockIdx.y * blockDim.x + threadIdx.x)
// ensures that only the upper-triangular portion is processed.
// This optimizes thread usage by reducing inactive threads and minimizing branch divergence.

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Each block in x covers one row
    int row = blockIdx.x;  
    // Compute the column offset for current row from blockIdx.y and threadIdx.x
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    int col = row + offset;  // valid if col < N
    
    if (col < N) {
        float sum = 0.0f;
        // Compute dot product over valid range [row, col]
        for (int k = row; k <= col; ++k) {
            sum = fmaf(A[row * N + k], B[k * N + col], sum);
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
// We launch a 2D grid where gridDim.x corresponds to the row index (ranging from 0 to N-1),
// and gridDim.y is chosen to cover the maximum possible number of valid column offsets (i.e. up to N elements).
// This tailored mapping minimizes wasted threads in the computation and improves runtime performance.

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Define block dimensions: using 32 threads along x for column offset processing.
    dim3 threadsPerBlock(32, 1);
    // gridDim.x = N for each row; gridDim.y is chosen to cover up to N columns in the worst-case (row 0)
    int grid_y = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    dim3 numBlocks(N, grid_y);
    
    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Efficient upper triangular matrix multiplication with dynamic thread mapping");
}
