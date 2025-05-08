#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor")

__inline__ __device__
float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += WARP_SIZE) {
            float partial_sum = 0.0f; 
            __syncwarp();
            #pragma unroll
            for (int w = 0; w < WARP_SIZE && (k + w) < TILE_SIZE; ++w) {
                partial_sum += As[threadIdx.y][k + w] * Bs[k + w][threadIdx.x];
            }
            sum += warpReduceSum(partial_sum);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyKernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with warp-level optimizations");
}