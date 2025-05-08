#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Kernel using double-buffering and asynchronous copies (cp.async) to overlap global memory loads with computation
__global__ void async_matmul_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    // Double-buffered shared memory for tiles from A and B
    __shared__ float s_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float value = 0.0f;

    // Total number of tiles needed (each tile is TILE_SIZE wide)
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int curr = 0;

    // Load the first tile asynchronously into shared memory
    if (numTiles > 0) {
        int aCol = 0 * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                          :
                          : "r"(&s_A[curr][ty][tx]), "l"(A + row * N + aCol), "n"(4));
        } else {
            s_A[curr][ty][tx] = 0.0f;
        }

        int bRow = 0 * TILE_SIZE + ty;
        if (col < N && bRow < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                          :
                          : "r"(&s_B[curr][ty][tx]), "l"(B + bRow * N + col), "n"(4));
        } else {
            s_B[curr][ty][tx] = 0.0f;
        }

        asm volatile ("cp.async.commit_group;");
        asm volatile ("cp.async.wait_group 0;");
        __syncthreads();
    }

    // For each subsequent tile, prefetch next tile and compute on current tile concurrently
    for (int tile = 1; tile < numTiles; tile++) {
        int next = curr ^ 1; // Toggle between 0 and 1 for double buffering

        int aCol = tile * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                          :
                          : "r"(&s_A[next][ty][tx]), "l"(A + row * N + aCol), "n"(4));
        } else {
            s_A[next][ty][tx] = 0.0f;
        }

        int bRow = tile * TILE_SIZE + ty;
        if (col < N && bRow < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                          :
                          : "r"(&s_B[next][ty][tx]), "l"(B + bRow * N + col), "n"(4));
        } else {
            s_B[next][ty][tx] = 0.0f;
        }

        asm volatile ("cp.async.commit_group;");

        // Compute on the current tile
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[curr][ty][k] * s_B[curr][k][tx];
        }

        // Wait for the async copies for the next tile to finish
        asm volatile ("cp.async.wait_group 0;");
        __syncthreads();
        curr = next;
    }

    // Process the final tile
    if (numTiles > 0) {
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[curr][ty][k] * s_B[curr][k][tx];
        }
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface that launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the asynchronous pipelined kernel
    async_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Asynchronous Pipelined Matrix Multiplication (CUDA)");
}
