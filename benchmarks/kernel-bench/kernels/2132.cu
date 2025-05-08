#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel uses double buffering to reduce the number of __syncthreads() calls.
// Two shared memory buffers (for A and B) are alternated between loading new tiles and computing with the previously loaded tile.
// This allows us to synchronize only once per tile iteration (after loading) rather than twice, reducing synchronization overhead.

__global__ void dp_buffered_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Declare double buffered shared memory for tiles of A and B
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int row = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
    int col = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);
    float sum = 0.0f;

    // For a lower triangular matrix, valid blocks satisfy blockIdx.y >= blockIdx.x
    int startTile = static_cast<int>(blockIdx.x);
    int endTile = static_cast<int>(blockIdx.y);

    if (row < N && col < N) {
        // current buffer index
        int current = 0;
        
        // Load the first tile (t = startTile) into the current buffer
        {
            int t = startTile;
            int a_col = t * TILE_SIZE + static_cast<int>(threadIdx.x);
            As[current][threadIdx.y][threadIdx.x] = (row < N && a_col < N && row >= a_col) 
                ? __ldg(&A[row * N + a_col]) : 0.0f;
            int b_row = t * TILE_SIZE + static_cast<int>(threadIdx.y);
            Bs[current][threadIdx.y][threadIdx.x] = (b_row < N && col < N && b_row >= col) 
                ? __ldg(&B[b_row * N + col]) : 0.0f;
        }
        __syncthreads(); // Ensure the first tile is loaded

        // Loop over remaining tiles using double buffering
        for (int t = startTile + 1; t <= endTile; ++t) {
            int next = 1 - current;  // alternate buffer
            // Load tile t into the next buffer
            {
                int a_col = t * TILE_SIZE + static_cast<int>(threadIdx.x);
                As[next][threadIdx.y][threadIdx.x] = (row < N && a_col < N && row >= a_col) 
                    ? __ldg(&A[row * N + a_col]) : 0.0f;
                int b_row = t * TILE_SIZE + static_cast<int>(threadIdx.y);
                Bs[next][threadIdx.y][threadIdx.x] = (b_row < N && col < N && b_row >= col) 
                    ? __ldg(&B[b_row * N + col]) : 0.0f;
            }
            __syncthreads(); // Synchronize to ensure the new tile is loaded
            
            // Compute using the previously loaded tile in the current buffer
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[current][threadIdx.y][k] * Bs[current][k][threadIdx.x];
            }

            // Swap buffers: the newly loaded tile becomes the current tile for next iteration
            current = next;
            // No additional __syncthreads() is needed here since the load synchronization above guarantees
            // that all threads have completed the tile load before computing the previous tile.
        }
        
        // Compute with the final loaded tile in the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[current][threadIdx.y][k] * Bs[current][k][threadIdx.x];
        }
    }
    
    // Write the computed value back to C if within bounds. For non-lower triangular parts, set to 0.
    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be the same size");

    int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    dp_buffered_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double buffered triangular matrix multiplication (CUDA)");
}
