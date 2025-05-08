#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block in the x-dimension corresponds to a single row of the matrix.
// The y-dimension of the grid splits the work across columns for that row.
// This mapping ensures that we only launch threads for valid indices in the lower triangular region,
// thus reducing wasted computations in the upper triangular area.

__global__ void triangular_mm_row_tile_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    // Each block in the x-direction corresponds to one row index.
    int row = blockIdx.x; 
    // The column index is computed from blockIdx.y and threadIdx.x
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (col <= row) {  // Lower triangular region
            float sum = 0.0f;
            #pragma unroll
            for (int k = col; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

// Host function exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Determine thread block configuration:
    // We'll use a 1D block for columns and assign each row to a block in the x-dimension.
    const int threads = 256; // number of threads to cover columns per tile
    int grid_y = (N + threads - 1) / threads; // number of column tiles required per row
    dim3 blocks(N, grid_y);
    dim3 threadsPerBlock(threads);

    triangular_mm_row_tile_kernel<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication using row-tiled thread indexing (CUDA)");
}
