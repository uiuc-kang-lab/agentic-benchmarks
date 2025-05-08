#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel leverages asynchronous copies (cp.async) for double buffering in shared memory
// to overlap global memory loads with computation. It uses two shared memory buffers for
// tiles of matrices A and B, preloading the next tile while computing on the current tile,
// reducing global memory latency and ensuring proper synchronization to avoid race conditions.

__global__ void matmul_double_buffer_async_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N) {
    // Double-buffered shared memory for A and B tiles
    __shared__ float A_tile[2][TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = blockRow * TILE_SIZE + ty;
    int col = blockCol * TILE_SIZE + tx;
    
    float c_val = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int currBuf = 0;  // current buffer index

    // Preload first tile (tile index = 0) into shared memory using asynchronous copy
    int tile = 0;
    {
        // Load tile for A
        int aCol = tile * TILE_SIZE + tx;
        int aRow = row;
        if(aRow < N && aCol < N) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(&A_tile[currBuf][ty][tx]), "l"(A + aRow * N + aCol), "n"(4));
        } else {
            A_tile[currBuf][ty][tx] = 0.0f;
        }
        
        // Load tile for B
        int bRow = tile * TILE_SIZE + ty;
        int bCol = col;
        if(bRow < N && bCol < N) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(&B_tile[currBuf][ty][tx]), "l"(B + bRow * N + bCol), "n"(4));
        } else {
            B_tile[currBuf][ty][tx] = 0.0f;
        }
    }
    // Ensure the first tile is loaded before computation
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    // Loop over all tiles
    for (tile = 0; tile < numTiles; tile++) {
        int nextTile = tile + 1;
        int nextBuf = 1 - currBuf;

        // Asynchronously preload the next tile if it exists
        if (nextTile < numTiles) {
            // Preload A for next tile
            int aCol = nextTile * TILE_SIZE + tx;
            int aRow = row;
            if(aRow < N && aCol < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                              :
                              : "r"(&A_tile[nextBuf][ty][tx]), "l"(A + aRow * N + aCol), "n"(4));
            } else {
                A_tile[nextBuf][ty][tx] = 0.0f;
            }
            
            // Preload B for next tile
            int bRow = nextTile * TILE_SIZE + ty;
            int bCol = col;
            if(bRow < N && bCol < N) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                              :
                              : "r"(&B_tile[nextBuf][ty][tx]), "l"(B + bRow * N + bCol), "n"(4));
            } else {
                B_tile[nextBuf][ty][tx] = 0.0f;
            }
        }
        // Wait for the async copies of current tile to complete
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Compute partial result using the current tile in shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += A_tile[currBuf][ty][k] * B_tile[currBuf][k][tx];
        }
        __syncthreads();

        // Swap buffers if next tile exists for the subsequent iteration
        if (nextTile < numTiles) {
            currBuf = nextBuf;
        }
    }

    // Write the result back to global memory
    if (row < N && col < N) {
        C[row * N + col] = c_val;
    }
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_double_buffer_async_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA)");
}
