#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_optimized_hybrid_kernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C, 
                                             const int N) {
    typedef float4 vec_t;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    if (row >= N || col >= N) return;

    float C_value = 0.0f;
    
    #pragma unroll 2
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if ((m * TILE_SIZE + tx < N) && ((row * N + m * TILE_SIZE + tx) % 4 == 0) && (tx % 4 == 0)) {
            vec_t tmp = *reinterpret_cast<const vec_t*>(&A[row * N + m * TILE_SIZE + tx]);
            As[ty][tx] = tmp.x;
            if (tx + 1 < TILE_SIZE) As[ty][tx + 1] = tmp.y;
            if (tx + 2 < TILE_SIZE) As[ty][tx + 2] = tmp.z;
            if (tx + 3 < TILE_SIZE) As[ty][tx + 3] = tmp.w;
        } else if (m * TILE_SIZE + tx < N) {
            As[ty][tx] = __ldg(&A[row * N + m * TILE_SIZE + tx]);
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((m * TILE_SIZE + ty < N) && (((m * TILE_SIZE + ty) * N + col) % 4 == 0) && (ty % 4 == 0)) {
            vec_t tmp = *reinterpret_cast<const vec_t*>(&B[(m * TILE_SIZE + ty) * N + col]);
            Bs[ty][tx] = tmp.x;
            if (ty + 1 < TILE_SIZE) Bs[ty + 1][tx] = tmp.y;
            if (ty + 2 < TILE_SIZE) Bs[ty + 2][tx] = tmp.z;
            if (ty + 3 < TILE_SIZE) Bs[ty + 3][tx] = tmp.w;
        } else if (m * TILE_SIZE + ty < N) {
            Bs[ty][tx] = __ldg(&B[(m * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value = __fmaf_rn(As[ty][k], Bs[k][tx], C_value);
        }

        __syncthreads();
    }

    C[row * N + col] = C_value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_optimized_hybrid_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA)");
}