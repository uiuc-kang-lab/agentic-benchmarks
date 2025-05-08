#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_DIM 128

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {
    
    __shared__ scalar_t As[TILE_DIM][TILE_DIM/2];
    __shared__ scalar_t Bs[TILE_DIM/2][TILE_DIM];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int batch_idx = blockIdx.z;
    const int row_start = blockIdx.x * TILE_DIM;
    const int col_start = blockIdx.y * TILE_DIM;
    
    // Each thread accumulates its own results
    scalar_t thread_results[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process K dimension in tiles
    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_DIM/2) {
        // Collaborative loading of A and B tiles into shared memory
        // Each thread loads multiple elements in a coalesced manner
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = row_start + (threadIdx.x / 2);
            int col = tile_idx + (threadIdx.x % 2) + i * (WARP_SIZE/2);
            
            if (row < M && col < K) {
                As[threadIdx.x / 2][threadIdx.x % 2 + i * (WARP_SIZE/2)] = 
                    A[batch_idx * M * K + row * K + col];
            }
            
            row = tile_idx + (threadIdx.x / 2);
            col = col_start + (threadIdx.x % 2) + i * (WARP_SIZE/2);
            
            if (row < K && col < L) {
                Bs[threadIdx.x / 2][threadIdx.x % 2 + i * (WARP_SIZE/2)] = 
                    B[row * L + col];
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_DIM/2; k++) {
            scalar_t a_val = As[warp_id * 4 + (lane_id/8)][k];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                thread_results[i] += a_val * 
                    Bs[k][lane_id % 8 + i * 8 + (lane_id/8) * 32];
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory in a coalesced manner
    const int out_row = row_start + warp_id * 4 + (lane_id/8);
    const int out_col_base = col_start + lane_id % 8;
    
    if (out_row < M) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int out_col = out_col_base + i * 8 + (lane_id/8) * 32;
            if (out_col < L) {
                output[batch_idx * M * L + out_row * L + out_col] = thread_results[i];
            }
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
    
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    dim3 threads(256);  // 8 warps per block
    dim3 blocks(
        (M + TILE_DIM - 1) / TILE_DIM,
        (L + TILE_DIM - 1) / TILE_DIM,
        N
    );
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));
}

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    
    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);
    
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}