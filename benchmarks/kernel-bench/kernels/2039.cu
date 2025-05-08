#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4

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
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            dst[threadIdx.y][threadIdx.x * 4 + i] = (col + i < N && row < N) ? src[base_idx + i] : 0.0f;
        }
    }
}

__global__ void unrolled_vectorized_triangular_mm(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col_base = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (row < col_base && col_base < N) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (col_base + i >= N) continue;
            C[row * N + col_base + i] = 0.0f;
        }
        return;
    }

    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        const int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        load_tile_vectorized(A, As, row, tile_start + threadIdx.x * 4, N, N);
        load_tile_vectorized(B, Bs, tile_start + threadIdx.y, col_base, N, N);
        
        __syncthreads();

        const int k_start = max(tile_start, col_base);
        const int k_end = min(tile_start + TILE_SIZE, row + 1);

        #pragma unroll
        for (int k = k_start; k < k_end; k++) {
            const int bs_idx = k - tile_start;
            const float a_val = As[threadIdx.y][bs_idx];
            
            sum.x += a_val * Bs[bs_idx][threadIdx.x * 4];
            sum.y += a_val * Bs[bs_idx][threadIdx.x * 4 + 1];
            sum.z += a_val * Bs[bs_idx][threadIdx.x * 4 + 2];
            sum.w += a_val * Bs[bs_idx][threadIdx.x * 4 + 3];
        }
        
        __syncthreads();
    }

    if (row < N) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int col = col_base + i;
            if (col < N && row >= col) {
                C[row * N + col] = ((float*)&sum)[i];
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

    dim3 block(TILE_SIZE/4, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    unrolled_vectorized_triangular_mm<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aggressively unrolled vectorized triangular matrix multiplication (CUDA)");
}
