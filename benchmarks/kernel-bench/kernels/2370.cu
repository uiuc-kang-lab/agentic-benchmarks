#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE_M = 32;
const int BLOCK_SIZE_N = 8;
const int TILE_SIZE = 32;

__global__ void matmul_transposed_kernel_balanced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Each thread computes 4 elements in the M dimension
    const int thread_elements = 4;
    float results[thread_elements] = {0.0f};

    const int row_base = by * BLOCK_SIZE_M + ty * thread_elements;
    const int col = bx * BLOCK_SIZE_N + tx;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int k_offset = tile * TILE_SIZE;

        // Collaborative loading using vectorized access where possible
        if (tx < TILE_SIZE/4) {
            float4* As_vec = reinterpret_cast<float4*>(&As[ty][tx*4]);
            const float4* A_vec = reinterpret_cast<const float4*>(
                &A[row_base * K + k_offset + tx*4]);
            if (row_base < M && (k_offset + tx*4 + 3) < K) {
                *As_vec = *A_vec;
            } else {
                As[ty][tx*4] = 0.0f;
                As[ty][tx*4+1] = 0.0f;
                As[ty][tx*4+2] = 0.0f;
                As[ty][tx*4+3] = 0.0f;
            }
        }

        if (ty < TILE_SIZE/4) {
            float4* Bs_vec = reinterpret_cast<float4*>(&Bs[ty*4][tx]);
            const float4* B_vec = reinterpret_cast<const float4*>(
                &B[col * K + k_offset + ty*4]);
            if (col < N && (k_offset + ty*4 + 3) < K) {
                *Bs_vec = *B_vec;
            } else {
                Bs[ty*4][tx] = 0.0f;
                Bs[ty*4+1][tx] = 0.0f;
                Bs[ty*4+2][tx] = 0.0f;
                Bs[ty*4+3][tx] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float bval = Bs[k][tx];
            #pragma unroll
            for (int i = 0; i < thread_elements; ++i) {
                results[i] += As[ty * thread_elements + i][k] * bval;
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < thread_elements; ++i) {
        const int row = row_base + i;
        if (row < M && col < N) {
            C[row * N + col] = results[i];
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M/4);
    dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
              (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    matmul_transposed_kernel_balanced<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}