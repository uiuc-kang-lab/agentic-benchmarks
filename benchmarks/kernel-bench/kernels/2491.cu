#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    constexpr int TILE_SIZE = 16;
    constexpr int VECTOR_SIZE = 4;  // Process 4 elements at once
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ scalar_t smemA[2][TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ scalar_t smemB[2][TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Register cache for partial results
    scalar_t sum = 0;
    
    // Registers for double buffering
    scalar_t regA[VECTOR_SIZE];
    scalar_t regB[VECTOR_SIZE];
    
    const int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int loadIdxA = tid / VECTOR_SIZE;
    const int loadIdxB = tid % (TILE_SIZE * TILE_SIZE / VECTOR_SIZE);
    
    // Double buffering indices
    int current_buffer = 0;
    int next_buffer = 1;
    
    // Prefetch first tile
    if (tid < TILE_SIZE * TILE_SIZE / VECTOR_SIZE) {
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            const int k_a = 0 * TILE_SIZE + loadIdxA;
            const int k_b = 0 * TILE_SIZE + loadIdxB * VECTOR_SIZE + v;
            
            if (k_a < K && row < M) {
                regA[v] = A[k_a * M + row];
            }
            if (k_b < K && col < N) {
                regB[v] = B[col * K + k_b];
            }
        }
        
        // Store to shared memory
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            smemA[0][loadIdxA][threadIdx.x] = regA[v];
            smemB[0][loadIdxB * VECTOR_SIZE + v][threadIdx.y] = regB[v];
        }
    }
    
    __syncthreads();
    
    // Main loop
    #pragma unroll 2
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load next tile while computing current one
        if (tid < TILE_SIZE * TILE_SIZE / VECTOR_SIZE && tile + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                const int k_a = (tile + 1) * TILE_SIZE + loadIdxA;
                const int k_b = (tile + 1) * TILE_SIZE + loadIdxB * VECTOR_SIZE + v;
                
                if (k_a < K && row < M) {
                    regA[v] = A[k_a * M + row];
                }
                if (k_b < K && col < N) {
                    regB[v] = B[col * K + k_b];
                }
            }
        }
        
        // Compute current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(smemA[current_buffer][k][threadIdx.x],
                           smemB[current_buffer][k][threadIdx.y],
                           sum);
        }
        
        __syncthreads();
        
        // Store prefetched data to shared memory for next iteration
        if (tid < TILE_SIZE * TILE_SIZE / VECTOR_SIZE && tile + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                smemA[next_buffer][loadIdxA][threadIdx.x] = regA[v];
                smemB[next_buffer][loadIdxB * VECTOR_SIZE + v][threadIdx.y] = regB[v];
            }
        }
        
        __syncthreads();
        
        // Swap buffers
        current_buffer = 1 - current_buffer;
        next_buffer = 1 - next_buffer;
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    constexpr int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose (CUDA)");
}