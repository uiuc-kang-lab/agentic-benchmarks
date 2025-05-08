#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 64
#define VECTOR_WIDTH 4

typedef float4 vec_type;

__device__ __forceinline__ void load_aligned_vec(const float* src, vec_type& vec, int idx) {
    vec = *reinterpret_cast<const vec_type*>(src + idx);
}

__global__ void vectorized_2d_block_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col_base = blockIdx.x * TILE_SIZE * VECTOR_WIDTH + threadIdx.x * VECTOR_WIDTH;

    if (col_base >= N || row >= N) return;
    
    vec_type sum_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    const int tile_count = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < tile_count; ++t) {
        const int tile_start = t * TILE_SIZE;
        if (tile_start > row) break;

        // Vectorized load of A tile (row-major)
        if (threadIdx.x < TILE_SIZE/VECTOR_WIDTH && threadIdx.y < TILE_SIZE) {
            vec_type a_vec;
            const int a_idx = row * N + tile_start + threadIdx.x*VECTOR_WIDTH;
            if (tile_start + threadIdx.x*VECTOR_WIDTH + VECTOR_WIDTH <= N)
                load_aligned_vec(A, a_vec, a_idx);
            else
                a_vec = {0.0f, 0.0f, 0.0f, 0.0f};
            
            for(int i=0; i<VECTOR_WIDTH; i++)
                As[threadIdx.y][threadIdx.x*VECTOR_WIDTH + i] = ((float*)&a_vec)[i];
        }

        // Vectorized load of B tile (column-major chunks)
        if (threadIdx.y < TILE_SIZE/VECTOR_WIDTH && threadIdx.x < TILE_SIZE) {
            vec_type b_vec;
            const int b_idx = (tile_start + threadIdx.y*VECTOR_WIDTH) * N + col_base;
            if (col_base + VECTOR_WIDTH <= N && tile_start + threadIdx.y*VECTOR_WIDTH <= row)
                load_aligned_vec(B, b_vec, b_idx);
            else
                b_vec = {0.0f, 0.0f, 0.0f, 0.0f};

            for(int i=0; i<VECTOR_WIDTH; i++)
                Bs[threadIdx.y*VECTOR_WIDTH + i][threadIdx.x] = ((float*)&b_vec)[i];
        }
        __syncthreads();

        const int k_min = max(tile_start, col_base);
        const int k_max = min(tile_start + TILE_SIZE, row + 1);

        for(int k=k_min; k<k_max; k++) {
            const float a = As[threadIdx.y][k - tile_start];
            const float b = Bs[k - tile_start][threadIdx.x];
            sum_vec.x += a * b;
        }
        __syncthreads();
    }

    // Vectorized store with lower-triangular mask
    if (row < N && col_base < N) {
        const bool valid[4] = {
            (col_base + 0) <= row,
            (col_base + 1) <= row,
            (col_base + 2) <= row,
            (col_base + 3) <= row
        };

        #pragma unroll
        for(int i=0; i<VECTOR_WIDTH; i++) {
            const int col = col_base + i;
            if (col < N) {
                C[row*N + col] = valid[i] ? ((float*)&sum_vec)[i] : 0.0f;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
               "Matrices must be square and equal size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // 2D block (32x8) gives 256 threads/block, aligned with H100 SM architecture
    dim3 block(TILE_SIZE/VECTOR_WIDTH, TILE_SIZE);
    dim3 grid((N + TILE_SIZE*VECTOR_WIDTH - 1)/(TILE_SIZE*VECTOR_WIDTH), 
              (N + TILE_SIZE - 1)/TILE_SIZE);

    vectorized_2d_block_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                               B.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized 2D block triangular matrix multiplication (CUDA)");
}