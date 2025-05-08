#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Persistent cuBLAS handle for larger matrix multiplications
static cublasHandle_t handle = nullptr;

// Double-buffered tiled matrix multiplication kernel with minimal synchronizations
__global__ void doublebuffer_tiled_matmul_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   const int M, const int N, const int K) {
    // Allocate two shared memory buffers for double buffering
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global row and column indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // Determine the number of tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float sum = 0.0f;

    // 'curr' indicates the buffer index currently used for computation
    int curr = 0;

    // Load the first tile (tile 0) into buffer 'curr'
    int tiled_col = 0 * TILE_SIZE + tx;
    int tiled_row = 0 * TILE_SIZE + ty;
    As[curr][ty][tx] = (row < M && tiled_col < K) ? A[row * K + tiled_col] : 0.0f;
    Bs[curr][ty][tx] = (tiled_row < K && col < N) ? B[tiled_row * N + col] : 0.0f;

    // Ensure the first tile is loaded
    __syncthreads();

    // Loop over all tiles except the last one using double buffering
    for (int tile = 0; tile < numTiles - 1; tile++) {
        int next = 1 - curr;  // Alternate buffer index
        int next_tile = tile + 1;
        int tiled_col_next = next_tile * TILE_SIZE + tx;
        int tiled_row_next = next_tile * TILE_SIZE + ty;
        
        // Prefetch the next tile into the alternate buffer
        As[next][ty][tx] = (row < M && tiled_col_next < K) ? A[row * K + tiled_col_next] : 0.0f;
        Bs[next][ty][tx] = (tiled_row_next < K && col < N) ? B[tiled_row_next * N + col] : 0.0f;
        
        // Synchronize to ensure the next tile is fully loaded
        __syncthreads();
        
        // Compute partial results using the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }
        
        // Swap buffers: the prefetched tile becomes the current tile for the next iteration
        curr = next;
        // No additional __syncthreads() here; the upcoming prefetch synchronization suffices
    }

    // Process the last tile (already loaded in buffer 'curr')
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[curr][ty][k] * Bs[curr][k][tx];
    }

    // Write the result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Unified matrix multiplication: uses the double-buffered kernel for small matrices and cuBLAS for larger ones
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Use the custom kernel for small matrices; otherwise, use cuBLAS
    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        doublebuffer_tiled_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    } else {
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device())
                       .requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);
    
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double-buffered hybrid matrix multiplication (CUDA)");
}
