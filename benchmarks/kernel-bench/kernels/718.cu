#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block dimensions and tile size for the K dimension
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define TILE_K 16

// Macros to check input validity
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined kernel: Uses shared memory tiling (from kernel 1) with improved block indexing and __ldg caching (from kernel 2).
// Each thread computes one element of C. The A and B tiles are loaded in parallel via multiple loads per thread.
__global__ void tiled_improved_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N) {
    // Compute global output indices
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    float cValue = 0.0f;

    // Allocate shared memory for the A and B tiles
    // A tile: dimensions BLOCK_SIZE_Y x TILE_K (8 x 16)
    // B tile: dimensions TILE_K x BLOCK_SIZE_X (16 x 32)
    __shared__ float As[BLOCK_SIZE_Y][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE_X];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Compute a linear thread id for load mapping
    int tid = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;  // range [0, BLOCK_SIZE_X*BLOCK_SIZE_Y), here 0..255

    for (int t = 0; t < numTiles; t++) {
        // ------------------------------
        // Load tile from A into shared memory
        // A tile has BLOCK_SIZE_Y * TILE_K elements (8*16 = 128 elements).
        int totalA = BLOCK_SIZE_Y * TILE_K;  
        if (tid < totalA) {
            int a_row = tid / TILE_K;
            int a_col = tid % TILE_K;
            int globalRow = blockIdx.y * BLOCK_SIZE_Y + a_row;
            int globalCol = t * TILE_K + a_col;
            As[a_row][a_col] = (globalRow < M && globalCol < K) ? __ldg(&A[globalRow * K + globalCol]) : 0.0f;
        }

        // ------------------------------
        // Load tile from B into shared memory
        // B tile has TILE_K * BLOCK_SIZE_X elements (16*32 = 512 elements).
        // Each thread loads 2 elements since 256 threads * 2 = 512 elements.
        int totalB = TILE_K * BLOCK_SIZE_X;
        int bIndex1 = tid;
        int bIndex2 = tid + BLOCK_SIZE_X * BLOCK_SIZE_Y;  // offset by the number of threads (256)

        if (bIndex1 < totalB) {
            int b_row = bIndex1 / BLOCK_SIZE_X;
            int b_col = bIndex1 % BLOCK_SIZE_X;
            int globalRow = t * TILE_K + b_row;
            int globalCol = blockIdx.x * BLOCK_SIZE_X + b_col;
            Bs[b_row][b_col] = (globalRow < K && globalCol < N) ? __ldg(&B[globalRow * N + globalCol]) : 0.0f;
        }
        if (bIndex2 < totalB) {
            int b_row = bIndex2 / BLOCK_SIZE_X;
            int b_col = bIndex2 % BLOCK_SIZE_X;
            int globalRow = t * TILE_K + b_row;
            int globalCol = blockIdx.x * BLOCK_SIZE_X + b_col;
            Bs[b_row][b_col] = (globalRow < K && globalCol < N) ? __ldg(&B[globalRow * N + globalCol]) : 0.0f;
        }

        __syncthreads();

        // ---------------
        // Multiply the two tiles
        // Each thread accumulates a partial sum for its output element
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write back the result if within matrix boundaries
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// Host function exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Use block dimensions optimized for modern GPUs (e.g., H100): 32 threads in x and 8 threads in y
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                 (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    tiled_improved_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined tiled matrix multiplication with improved indexing and caching");
}
