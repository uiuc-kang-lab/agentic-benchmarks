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

__global__ void matmul_optimized_vectorized_kernel(const float* __restrict__ A, 
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
    
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        const int A_col = m * TILE_SIZE + tx;
        const int B_row = m * TILE_SIZE + ty;

        if (row < N && A_col < N) {
            float4 A_vec = reinterpret_cast<const float4*>(&A[row * N + m * TILE_SIZE])[tx / 4];
            As[ty][tx] = ((float*)&A_vec)[tx % 4];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (B_row < N && col < N) {
            float4 B_vec = reinterpret_cast<const float4*>(&B[B_row * N])[col / 4];
            Bs[ty][tx] = ((float*)&B_vec)[tx % 4];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k += 4) {
            float4 a_vec, b_vec;
            
            a_vec.x = As[ty][k];
            a_vec.y = As[ty][k+1];
            a_vec.z = As[ty][k+2];
            a_vec.w = As[ty][k+3];
            
            b_vec.x = Bs[k][tx];
            b_vec.y = Bs[k+1][tx];
            b_vec.z = Bs[k+2][tx];
            b_vec.w = Bs[k+3][tx];
            
            C_vec.x += a_vec.x * b_vec.x + a_vec.y * b_vec.y;
            C_vec.y += a_vec.z * b_vec.z + a_vec.w * b_vec.w;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = C_vec.x + C_vec.y;
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
    TORCH_CHECK(A.size(0) % 4 == 0, "Matrix dimension must be divisible by 4");

    const int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_optimized_vectorized_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA)");
}