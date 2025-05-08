#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4
#define NUM_STREAMS 4

__device__ __forceinline__ void load_tile_vectorized(const float* __restrict__ src,
                                                     float dst[TILE_SIZE][TILE_SIZE+1],
                                                     const int row, const int col,
                                                     const int N, const int stride) {
    float4 vec;
    int base_idx = row * stride + col;
    
    if (col + 4 <= N && row < N) {
        vec = *reinterpret_cast<const float4*>(&src[base_idx]);
        dst[threadIdx.y][threadIdx.x * 4] = vec.x;
        dst[threadIdx.y][threadIdx.x * 4 + 1] = vec.y;
        dst[threadIdx.y][threadIdx.x * 4 + 2] = vec.z;
        dst[threadIdx.y][threadIdx.x * 4 + 3] = vec.w;
    } else {
        for (int i = 0; i < 4; i++) {
            if (col + i < N && row < N) {
                dst[threadIdx.y][threadIdx.x * 4 + i] = src[base_idx + i];
            } else {
                dst[threadIdx.y][threadIdx.x * 4 + i] = 0.0f;
            }
        }
    }
}

__global__ void streamed_vector_triangular_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 const int N, const int start_row) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    const int row = start_row + blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col_base = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

    if (row >= N) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (col_base >= N || row < col_base) {
        if (row < col_base) { // Handle upper triangle initialization first
            for (int i = 0; i < 4; i++) {
                if (col_base + i < N) {
                    C[row * N + col_base + i] = 0.0f;
                }
            }
        }
        return;
    }

    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        const int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        // Load tiles using vectorized loads where possible
        load_tile_vectorized(A, As, row, tile_start + threadIdx.x * 4, N, N);
        load_tile_vectorized(B, Bs, tile_start + threadIdx.y, col_base, N, N);
        
        __syncthreads();

        const int k_start = max(tile_start, col_base);
        const int k_end = min(tile_start + TILE_SIZE, row + 1);

        #pragma unroll 8
        for (int k = k_start; k < k_end; k++) {
            const float a_val = As[threadIdx.y][k - tile_start];
            const int bs_idx = k - tile_start;
            
            sum.x += a_val * Bs[bs_idx][threadIdx.x * 4];
            sum.y += a_val * Bs[bs_idx][threadIdx.x * 4 + 1];
            sum.z += a_val * Bs[bs_idx][threadIdx.x * 4 + 2];
            sum.w += a_val * Bs[bs_idx][threadIdx.x * 4 + 3];
        }
        
        __syncthreads();
    }

    if (col_base < N) {
        for (int i = 0; i < 4; i++) {
            const int col = col_base + i;
            if (col < N && row >= col) {
                float result;
                switch(i) {
                    case 0: result = sum.x; break;
                    case 1: result = sum.y; break;
                    case 2: result = sum.z; break;
                    case 3: result = sum.w; break;
                }
                C[row * N + col] = result;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Create multiple CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int s = 0; s < NUM_STREAMS; s++) {
        const int start_row = s * chunk_size;
        const int rows_in_chunk = (s == NUM_STREAMS - 1) ? N - start_row : chunk_size;
        if (rows_in_chunk <= 0) break;

        dim3 block(TILE_SIZE/4, TILE_SIZE);
        dim3 grid((N + TILE_SIZE -1)/TILE_SIZE,
                  (rows_in_chunk + TILE_SIZE - 1)/TILE_SIZE);

        streamed_vector_triangular_kernel<<<grid, block, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
        );
    }

    // Sync and destroy streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-optimized vectorized triangular matmul (CUDA)");
}