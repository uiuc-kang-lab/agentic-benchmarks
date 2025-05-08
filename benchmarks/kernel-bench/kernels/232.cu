#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size for shared memory tiling
#define TILE_SIZE 16

// Macro to perform asynchronous copy from global to shared memory
// Copies 'bytes' bytes from src to dst using cp.async instruction
#define CP_ASYNC_SHARED_GLOBAL(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" : : "r"(dst), "l"(src), "n"(bytes) : "memory")

// Kernel using asynchronous double buffering to overlap global load with computation
__global__ void bmm_async_double_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Determine batch index from grid.z
    int b = blockIdx.z;
    
    // Calculate row and column for output matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;

    // Allocate double buffers in shared memory for A and B tiles
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    // Total number of tiles required to cover dimension K
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Use double buffering: 0 and 1 index
    int curr_buffer = 0;

    // Preload first tile into the current buffer using asynchronous copy
    {
        int t = 0;
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        
        if(row < M && aCol < K) {
            CP_ASYNC_SHARED_GLOBAL(&As[curr_buffer][threadIdx.y][threadIdx.x],
                &A[b * M * K + row * K + aCol], sizeof(float));
        } else {
            As[curr_buffer][threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(bRow < K && col < N) {
            CP_ASYNC_SHARED_GLOBAL(&Bs[curr_buffer][threadIdx.y][threadIdx.x],
                &B[b * K * N + bRow * N + col], sizeof(float));
        } else {
            Bs[curr_buffer][threadIdx.y][threadIdx.x] = 0.0f;
        }
    }

    // Ensure the first asynchronous copies are completed
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Loop over remaining tiles with double buffering
    for (int t = 1; t < numTiles; t++) {
        int next_buffer = 1 - curr_buffer;

        // Launch asynchronous copy for the next tile into the next buffer
        {
            int aCol = t * TILE_SIZE + threadIdx.x;
            int bRow = t * TILE_SIZE + threadIdx.y;
            
            if(row < M && aCol < K) {
                CP_ASYNC_SHARED_GLOBAL(&As[next_buffer][threadIdx.y][threadIdx.x],
                    &A[b * M * K + row * K + aCol], sizeof(float));
            } else {
                As[next_buffer][threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if(bRow < K && col < N) {
                CP_ASYNC_SHARED_GLOBAL(&Bs[next_buffer][threadIdx.y][threadIdx.x],
                    &B[b * K * N + bRow * N + col], sizeof(float));
            } else {
                Bs[next_buffer][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        // Wait for the asynchronous copies of the next tile to complete
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        
        // Compute the current tile using data in the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += As[curr_buffer][threadIdx.y][k] * Bs[curr_buffer][k][threadIdx.x];
        }
        
        // Swap buffers
        curr_buffer = next_buffer;
        __syncthreads();
    }

    // Final computation on the last loaded tile
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        acc += As[curr_buffer][threadIdx.y][k] * Bs[curr_buffer][k][threadIdx.x];
    }
    
    // Write the final result
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = acc;
    }
}

// Forward function to launch the asynchronous double-buffered kernel
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
    
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({batch_size, M, N}, options);
    
    // Configure block and grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    bmm_async_double_buffer_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with asynchronous double buffering (CUDA)");
}
