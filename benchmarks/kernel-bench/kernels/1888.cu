#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define CHUNK_SIZE 4  // Aligned with float4 size

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE + 4]; // +4 for bank conflict avoidance
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE + 4];
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Register array for accumulation aligned to float4
    float reg_C[CHUNK_SIZE] __attribute__((aligned(16))) = {0.0f};
    
    // Loop over block-level tiles
    for (int bk = 0; bk <= (row / BLOCK_SIZE); bk++) {
        const int block_start = bk * BLOCK_SIZE;
        
        // Vectorized loading using float4
        if (row < N && (block_start + threadIdx.x * 4) < N) {
            float4 tmp_A = *reinterpret_cast<const float4*>(&__ldg(&A[row * N + block_start + threadIdx.x * 4]));
            s_A[threadIdx.y][threadIdx.x * 4] = tmp_A.x;
            s_A[threadIdx.y][threadIdx.x * 4 + 1] = tmp_A.y;
            s_A[threadIdx.y][threadIdx.x * 4 + 2] = tmp_A.z;
            s_A[threadIdx.y][threadIdx.x * 4 + 3] = tmp_A.w;
        } else {
            s_A[threadIdx.y][threadIdx.x * 4] = 0.0f;
            s_A[threadIdx.y][threadIdx.x * 4 + 1] = 0.0f;
            s_A[threadIdx.y][threadIdx.x * 4 + 2] = 0.0f;
            s_A[threadIdx.y][threadIdx.x * 4 + 3] = 0.0f;
        }
        
        if ((block_start + threadIdx.y) < N && col < N) {
            float4 tmp_B = *reinterpret_cast<const float4*>(&__ldg(&B[(block_start + threadIdx.y) * N + col - (col % 4)]));
            s_B[threadIdx.y][threadIdx.x] = tmp_B.x + (col % 4 == 0 ? 0 : (col % 4 == 1 ? tmp_B.y : (col % 4 == 2 ? tmp_B.z : tmp_B.w)));
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        if (row < N && col < N && row >= col) {
            const int k_start = max(block_start, col);
            const int k_end = min(block_start + BLOCK_SIZE, row + 1);
            
            #pragma unroll
            for (int k = k_start; k < k_end; k += CHUNK_SIZE) {
                float4 a_vec = *reinterpret_cast<float4*>(&s_A[threadIdx.y][k - block_start]);
                float4 b_vec = *reinterpret_cast<float4*>(&s_B[k - block_start][threadIdx.x]);
                
                #pragma unroll
                for (int c = 0; c < CHUNK_SIZE && (k + c) < k_end; c++) {
                    reg_C[c] += a_vec.x * b_vec.x + a_vec.y * b_vec.y + 
                               a_vec.z * b_vec.z + a_vec.w * b_vec.w;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Aligned store using float4
    if (row < N && col < N) {
        if (row >= col) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; i++) {
                sum += reg_C[i];
            }
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}