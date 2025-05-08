#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4  // Process 4 elements at once using float4

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void matmul_tiled_vectorized_kernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float4 C_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 2
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        const int A_col = m * TILE_SIZE + tx;
        const int B_row = m * TILE_SIZE + ty;

        if (row < N && A_col + VECTOR_SIZE <= N) {
            float4 A_vec = load_float4(&A[row * N + A_col]);
            As[ty][tx] = A_vec.x;
            if (tx + 1 < TILE_SIZE) As[ty][tx + 1] = A_vec.y;
            if (tx + 2 < TILE_SIZE) As[ty][tx + 2] = A_vec.z;
            if (tx + 3 < TILE_SIZE) As[ty][tx + 3] = A_vec.w;
        } else if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (B_row < N && col + VECTOR_SIZE <= N) {
            float4 B_vec = load_float4(&B[B_row * N + col]);
            Bs[ty][tx] = B_vec.x;
            if (ty + 1 < TILE_SIZE) Bs[ty + 1][tx] = B_vec.y;
            if (ty + 2 < TILE_SIZE) Bs[ty + 2][tx] = B_vec.z;
            if (ty + 3 < TILE_SIZE) Bs[ty + 3][tx] = B_vec.w;
        } else if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += VECTOR_SIZE) {
            float4 A_vec = load_float4(&As[ty][k]);
            float4 B_vec = load_float4(&Bs[k][tx]);
            
            C_vec.x += A_vec.x * B_vec.x + A_vec.y * B_vec.y + 
                       A_vec.z * B_vec.z + A_vec.w * B_vec.w;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = C_vec.x + C_vec.y + C_vec.z + C_vec.w;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");
    TORCH_CHECK(A.size(0) % VECTOR_SIZE == 0, "Matrix dimension must be divisible by vector size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_vectorized_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}