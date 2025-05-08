#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define ALIGN_MASK 0xFFFFFFF0  // 16-byte alignment mask

__forceinline__ __device__ float4 load_float4_aligned(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    __shared__ float shB[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const int t_start = col / TILE_SIZE;
    const int t_end = row / TILE_SIZE;

    #pragma unroll 4
    for (int t = t_start; t <= t_end; t++) {
        // Aligned vector loads for tile A
        if (threadIdx.x < TILE_SIZE/4) {
            int base_idx = row * N + t * TILE_SIZE + threadIdx.x * 4;
            if (base_idx + 3 < N && (t * TILE_SIZE + threadIdx.x * 4) <= row) {
                float4 va = load_float4_aligned(A + (base_idx & ALIGN_MASK));
                shA[threadIdx.y][threadIdx.x * 4] = va.x;
                shA[threadIdx.y][threadIdx.x * 4 + 1] = va.y;
                shA[threadIdx.y][threadIdx.x * 4 + 2] = va.z;
                shA[threadIdx.y][threadIdx.x * 4 + 3] = va.w;
            } else {
                #pragma unroll 4
                for (int i = 0; i < 4; i++) {
                    int a_col = t * TILE_SIZE + threadIdx.x * 4 + i;
                    shA[threadIdx.y][threadIdx.x * 4 + i] = 
                        (a_col < N && a_col <= row) ? __ldg(&A[row * N + a_col]) : 0.0f;
                }
            }
        }

        // Aligned vector loads for tile B
        if (threadIdx.y < TILE_SIZE/4) {
            int base_idx = (t * TILE_SIZE + threadIdx.y * 4) * N + col;
            if (base_idx + 3*N < N*N && (t * TILE_SIZE + threadIdx.y * 4) >= col) {
                #pragma unroll 4
                for (int i = 0; i < 4; i++) {
                    shB[threadIdx.y * 4 + i][threadIdx.x] = 
                        __ldg(&B[base_idx + i * N]);
                }
            } else {
                #pragma unroll 4
                for (int i = 0; i < 4; i++) {
                    int b_row = t * TILE_SIZE + threadIdx.y * 4 + i;
                    shB[threadIdx.y * 4 + i][threadIdx.x] = 
                        (b_row < N && b_row >= col) ? __ldg(&B[b_row * N + col]) : 0.0f;
                }
            }
        }

        __syncthreads();

        int k_start = max(t * TILE_SIZE, col);
        int k_end = min((t + 1) * TILE_SIZE, row + 1);

        if (k_end - k_start == TILE_SIZE) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k += 4) {
                sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
                sum += shA[threadIdx.y][k+1] * shB[k+1][threadIdx.x];
                sum += shA[threadIdx.y][k+2] * shB[k+2][threadIdx.x];
                sum += shA[threadIdx.y][k+3] * shB[k+3][threadIdx.x];
            }
        } else {
            for (int k = k_start - t * TILE_SIZE; k < k_end - t * TILE_SIZE; k++) {
                sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferL1);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA) with aligned memory access");
}