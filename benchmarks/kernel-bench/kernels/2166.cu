#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4  // Using float4 for vectorized loads

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    
    // Early exit for upper triangle
    if (row < col) {
        if ((col & 3) == 0 && col + 3 < N) {
            float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
            store_float4(&C[row * N + col], zeros);
        } else {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    // Initialize accumulator registers
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Compute tile boundaries
    const int t_start = col / TILE_SIZE;
    const int t_end = row / TILE_SIZE;

    #pragma unroll 4
    for (int t = t_start; t <= t_end; t++) {
        // Vector load A tile when possible
        if (threadIdx.x < (TILE_SIZE/4)) {
            int a_col = t * TILE_SIZE + threadIdx.x * 4;
            if (a_col + 3 < N && a_col + 3 <= row) {
                float4 vecA = load_float4(&A[row * N + a_col]);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    shA[threadIdx.y][threadIdx.x * 4 + i] = ((float*)&vecA)[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int curr_col = a_col + i;
                    shA[threadIdx.y][threadIdx.x * 4 + i] = 
                        (curr_col < N && curr_col <= row) ? A[row * N + curr_col] : 0.0f;
                }
            }
        }

        // Vector load B tile when possible
        if (threadIdx.y < (TILE_SIZE/4)) {
            int b_row = t * TILE_SIZE + threadIdx.y * 4;
            if (b_row + 3 < N && b_row >= col) {
                float4 vecB = load_float4(&B[b_row * N + col]);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    shB[threadIdx.y * 4 + i][threadIdx.x] = ((float*)&vecB)[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int curr_row = b_row + i;
                    shB[threadIdx.y * 4 + i][threadIdx.x] = 
                        (curr_row < N && curr_row >= col) ? B[curr_row * N + col] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute local tile multiplication with aggressive unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float a_val = shA[threadIdx.y][k + i];
                float b_val = shB[k + i][threadIdx.x];
                acc[i] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Reduce accumulated values
    float final_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        final_sum += acc[i];
    }

    C[row * N + col] = final_sum;
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

    // Set L1 cache preference
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
    m.def("forward", &forward, "Aggressively unrolled triangular matrix multiplication (CUDA)");
}