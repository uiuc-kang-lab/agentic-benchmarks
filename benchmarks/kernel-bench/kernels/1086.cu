#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 16
#define BLOCK_ROWS 8
#define ELEMENTS_PER_THREAD 4

template <typename scalar_t>
__global__ void shared_memory_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {
    
    // Shared memory declarations with padding to avoid bank conflicts
    __shared__ scalar_t smem_A[2][TILE_DIM * (TILE_DIM + 1)];
    __shared__ scalar_t smem_B[2][TILE_DIM * (TILE_DIM + 1)];
    
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Register array to store partial results
    scalar_t thread_results[ELEMENTS_PER_THREAD] = {0};
    
    // Offset to the current batch
    const scalar_t* batch_A = A + batch_idx * M * K;
    scalar_t* batch_output = output + batch_idx * M * L;
    
    // Double-buffering indices
    int current_buffer = 0;
    
    // Preload first tile
    if (row < M && threadIdx.x < TILE_DIM) {
        smem_A[0][threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = 
            (row < M && threadIdx.x < K) ? batch_A[row * K + threadIdx.x] : 0;
    }
    if (threadIdx.y < TILE_DIM && col < L) {
        smem_B[0][threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = 
            (threadIdx.y < K && col < L) ? B[threadIdx.y * L + col] : 0;
    }
    __syncthreads();
    
    // Main loop over tiles
    #pragma unroll 4
    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Load next tile while computing current one
        if (tile + 1 < (K + TILE_DIM - 1) / TILE_DIM) {
            const int next_tile_idx = (tile + 1) * TILE_DIM;
            if (row < M && threadIdx.x < TILE_DIM) {
                smem_A[1 - current_buffer][threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = 
                    (row < M && next_tile_idx + threadIdx.x < K) ? 
                    batch_A[row * K + next_tile_idx + threadIdx.x] : 0;
            }
            if (threadIdx.y < TILE_DIM && col < L) {
                smem_B[1 - current_buffer][threadIdx.y * (TILE_DIM + 1) + threadIdx.x] = 
                    (next_tile_idx + threadIdx.y < K && col < L) ? 
                    B[(next_tile_idx + threadIdx.y) * L + col] : 0;
            }
        }
        
        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            const scalar_t a_val = smem_A[current_buffer][threadIdx.y * (TILE_DIM + 1) + k];
            const scalar_t b_val = smem_B[current_buffer][k * (TILE_DIM + 1) + threadIdx.x];
            
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                if (row + i * BLOCK_ROWS < M) {
                    thread_results[i] += a_val * b_val;
                }
            }
        }
        
        current_buffer = 1 - current_buffer;
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (row + i * BLOCK_ROWS < M && col < L) {
            batch_output[(row + i * BLOCK_ROWS) * L + col] = thread_results[i];
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
    
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);
    
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM,
              (M + BLOCK_ROWS * ELEMENTS_PER_THREAD - 1) / (BLOCK_ROWS * ELEMENTS_PER_THREAD),
              N);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "shared_memory_optimized_kernel", ([&] {
        shared_memory_optimized_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);
    
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared Memory Optimized 3D Matrix Multiplication (CUDA)");
}