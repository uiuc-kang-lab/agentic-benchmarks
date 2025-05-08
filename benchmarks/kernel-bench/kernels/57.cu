#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_warp_shuffle_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         const int N) {
    const int warpId = threadIdx.y / (WARP_SIZE/BLOCK_SIZE);
    const int laneId = threadIdx.x + (threadIdx.y % (WARP_SIZE/BLOCK_SIZE)) * BLOCK_SIZE;
    
    const int row = blockIdx.y * TILE_DIM + threadIdx.y * 2;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x * 2;

    float regC[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    const unsigned int FULL_MASK = 0xffffffff;

    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; t++) {
        // Load A into shared memory
        for (int i = 0; i < 2; i++) {
            int r = row + i;
            int c = t * TILE_DIM + threadIdx.x;
            if (r < N && c < N) {
                As[threadIdx.y * 2 + i][threadIdx.x] = A[r * N + c];
            } else {
                As[threadIdx.y * 2 + i][threadIdx.x] = 0.0f;
            }
        }
        // Load B into shared memory
        for (int j = 0; j < 2; j++) {
            int r = t * TILE_DIM + threadIdx.y;
            int c = col + j;
            if (r < N && c < N) {
                Bs[threadIdx.y][threadIdx.x * 2 + j] = B[r * N + c];
            } else {
                Bs[threadIdx.y][threadIdx.x * 2 + j] = 0.0f;
            }
        }
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a0 = As[threadIdx.y * 2][k];
            float a1 = As[threadIdx.y * 2 + 1][k];
            float b0 = Bs[k][threadIdx.x * 2];
            float b1 = Bs[k][threadIdx.x * 2 + 1];

            #pragma unroll
            for (int w = 0; w < WARP_SIZE/2; w++) {
                float a0_shared = __shfl_sync(FULL_MASK, a0, w, WARP_SIZE);
                float a1_shared = __shfl_sync(FULL_MASK, a1, w, WARP_SIZE);
                float b0_shared = __shfl_sync(FULL_MASK, b0, w, WARP_SIZE);
                float b1_shared = __shfl_sync(FULL_MASK, b1, w, WARP_SIZE);

                regC[0][0] += a0_shared * b0_shared;
                regC[0][1] += a0_shared * b1_shared;
                regC[1][0] += a1_shared * b0_shared;
                regC[1][1] += a1_shared * b1_shared;
            }
        }
        
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = regC[0][0];
        if (col + 1 < N) C[row * N + col + 1] = regC[0][1];
        if (row + 1 < N) {
            C[(row + 1) * N + col] = regC[1][0];
            if (col + 1 < N) C[(row + 1) * N + col + 1] = regC[1][1];
        }
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

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    matmul_warp_shuffle_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel with warp shuffle (CUDA)");
}