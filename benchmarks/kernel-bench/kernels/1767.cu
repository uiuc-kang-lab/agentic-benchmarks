#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size for the output and the segmentation along the k-dimension
#define BLOCK_SIZE_OUT 16
#define SEG_SIZE 32

// This kernel partitions the dot-product (summation over k) for each output element into segments.
// The grid is 3D: 
//   - gridDim.x and gridDim.y cover tiles of the output matrix C.
//   - gridDim.z partitions the summation (k index) into segments of length SEG_SIZE.
// For each output element C[i, j] in the lower triangular region (i >= j), the valid summation is over k from j to i.
// In a given k-segment, the contribution is computed as the sum over k in the intersection: [max(j, k_start), min(i, k_end)].
// Each thread in the block processes a portion of that segment, and a reduction in shared memory is performed
// so that a single atomicAdd updates C[i, j]. Atomic operations are used only once per block to accumulate
// a partial sum, thereby minimizing contention in global memory.

__global__ void triangular_mm_atomic_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    // Determine the output tile coordinates
    int tileRow = blockIdx.y * BLOCK_SIZE_OUT;
    int tileCol = blockIdx.x * BLOCK_SIZE_OUT;

    // Global row and column for this thread within the tile
    int row = tileRow + threadIdx.y;
    int col = tileCol + threadIdx.x;

    // Each block in the z-dimension handles a segment of the k-index
    int seg = blockIdx.z;
    int k_start = seg * SEG_SIZE;
    int k_end = k_start + SEG_SIZE - 1;
    if (k_end >= N) k_end = N - 1;

    // Only compute for valid indices: within matrix bounds and for lower triangular region (row >= col)
    if (row < N && col < N && row >= col) {
        float temp = 0.0f;
        // The valid summation for C[row, col] is over k in [col, row].
        // For this segment, we compute the intersection with [k_start, k_end]: from max(col, k_start) to min(row, k_end).
        int k_begin = (col > k_start) ? col : k_start;
        int k_final = (row < k_end) ? row : k_end;

        // If k_begin > k_final then there is no work in this segment
        for (int k = k_begin; k <= k_final; k++) {
            temp += A[row * N + k] * B[k * N + col];
        }

        // Use atomicAdd to accumulate the partial result from this k-segment block into global memory
        atomicAdd(&C[row * N + col], temp);
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
    // Create output tensor C and initialize to zero so that atomic adds work correctly
    auto C = torch::zeros_like(A);

    // Calculate grid dimensions
    // gridDim.x and gridDim.y tile the output matrix. We cover all indices [0, N) in each dimension.
    int grid_x = (N + BLOCK_SIZE_OUT - 1) / BLOCK_SIZE_OUT;
    int grid_y = (N + BLOCK_SIZE_OUT - 1) / BLOCK_SIZE_OUT;
    // gridDim.z partitions the k-dimension. The maximum possible summation length is N, so we use N/SEG_SIZE segments
    int grid_z = (N + SEG_SIZE - 1) / SEG_SIZE;
    
    dim3 blocks(grid_x, grid_y, grid_z);
    dim3 threads(BLOCK_SIZE_OUT, BLOCK_SIZE_OUT);

    // Launch the kernel
    triangular_mm_atomic_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Lower triangular matrix multiplication with atomic partial sums (CUDA)");
}
