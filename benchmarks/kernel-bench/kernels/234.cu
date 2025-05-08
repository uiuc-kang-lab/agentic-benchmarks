#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size and alignment mask
#define TILE_SIZE 16
#define ALIGN_MASK 0x0000000F

// Optimized CUDA kernel combining shared memory tiling with memory alignment and padding to avoid bank conflicts
__global__ void bmm_aligned_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Batch index from grid's z-dimension
    int b = blockIdx.z;

    // Compute row and column indices for the output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Allocate shared memory with an extra column to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Set up pointers for the current batch
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C = C + b * M * N;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over the tiles
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE_SIZE + threadIdx.x;
        int rowB = t * TILE_SIZE + threadIdx.y;

        // Load tile from matrix A into shared memory using __ldg for potential improved caching in read-only loads
        if (row < M && colA < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&batch_A[row * K + colA]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from matrix B
        if (rowB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&batch_B[rowB * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (row < M && col < N) {
        batch_C[row * N + col] = sum;
    }
}

// Forward function that sets up and launches the kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Ensure that input tensors are 16-byte aligned for optimal __ldg usage
    TORCH_CHECK((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) & ALIGN_MASK) == 0,
                "Input tensor A must be 16-byte aligned");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(B.data_ptr<float>()) & ALIGN_MASK) == 0,
                "Input tensor B must be 16-byte aligned");

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Configure block and grid dimensions based on the tile size
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    bmm_aligned_tiled_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Optimized batched matrix multiplication combining tiling, alignment, and padding (CUDA)");
}
