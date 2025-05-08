#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel assigns one warp per output element in the lower triangular part of C.
// Each warp collaboratively computes sum_{k=col}^{row} A[row, k] * B[k, col] using warp-level reduction.
// No atomic operations on global memory are used because each output element is handled by exactly one warp.

__global__ void triangular_mm_warp_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N,
                                            int M) {
    // Compute the global warp ID. Each block has (blockDim.x / 32) warps.
    int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31; // index within the warp

    if (warp_id < M) {
        // Map the linear warp_id (in lower triangular space) to (row, col) indices.
        // We solve for row in: row*(row+1)/2 <= warp_id < (row+1)*(row+2)/2
        float f_wid = (float)warp_id;
        float temp = sqrtf(8.f * f_wid + 1.f);
        int row = (int)floorf((temp - 1.f) * 0.5f);
        int row_start = row * (row + 1) / 2;
        int col = warp_id - row_start;

        if (row < N && col < N) {
            float sum = 0.f;
            // Each lane in the warp processes part of the summation range [col, row].
            for (int k = col + lane; k <= row; k += 32) {
                sum += A[row * N + k] * B[k * N + col];
            }
            // Warp-level reduction using shuffle instructions
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            // Only lane 0 writes the result
            if (lane == 0) {
                C[row * N + col] = sum;
            }
        }
    }
}

// The forward function initializes the output matrix and launches the warp-level kernel
// Only the lower triangular part is computed; the upper triangular part is set to zero via cudaMemset
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
    // Initialize the entire output matrix to zero (this sets the upper triangular part to zero)
    cudaMemset(C.data_ptr<float>(), 0, N * N * sizeof(float));

    // Compute the number of lower triangular elements
    int M = N * (N + 1) / 2;
    // Configure kernel: use blockDim.x as 128 (a multiple of warpSize) so each block has 4 warps.
    int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

    triangular_mm_warp_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with warp-level reduction and minimal atomic operations (CUDA)");
}
