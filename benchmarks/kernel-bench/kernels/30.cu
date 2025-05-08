#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define BM 4
#define BN 8
#define KT 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void optimized_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    if (row >= N || col >= N) return;

    float C_value = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (m * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + m * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (m * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        float partial = As[ty][tx] * Bs[ty][tx];

        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        
        if(tx % warpSize == 0) {
            C_value += partial;
        }

        __syncthreads();
    }

    if(tx % warpSize == 0) {
        C[row * N + col] = C_value;
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

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication kernel (CUDA)");
}