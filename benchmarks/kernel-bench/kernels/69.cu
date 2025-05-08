#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel uses vectorized loads via __ldg() with float4 to enforce 128-bit aligned memory accesses
// for the read-only matrices A and B. It loads tiles of A and B into shared memory using float4 loads
// when possible (for threads where threadIdx.x % 4 == 0) and falls back to scalar loads at boundaries.

__global__ void matmul_vectorized_aligned_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // 0 .. 31
    int ty = threadIdx.y; // 0 .. 31
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < numTiles; m++) {
        int aColStart = m * TILE_SIZE;
        int bRowStart = m * TILE_SIZE;

        // Load A tile into shared memory
        if (row < N) {
            // Use vectorized load for groups of 4 elements when possible
            if ((tx & 3) == 0) {  // equivalent to (tx % 4 == 0)
                int globalCol = aColStart + tx;
                if (globalCol <= N - 4) {
                    // reinterpret as float4 pointer for 128-bit load
                    const float4* A_vec = reinterpret_cast<const float4*>(A);
                    int index = row * N + aColStart + tx; // index in float elements
                    float4 a_val = __ldg(&A_vec[index / 4]);
                    As[ty][tx]     = a_val.x;
                    As[ty][tx + 1] = a_val.y;
                    As[ty][tx + 2] = a_val.z;
                    As[ty][tx + 3] = a_val.w;
                } else {
                    // Fallback scalar loads
                    for (int i = 0; i < 4; i++) {
                        int col_idx = aColStart + tx + i;
                        As[ty][tx + i] = (col_idx < N) ? __ldg(&A[row * N + col_idx]) : 0.0f;
                    }
                }
            }
        } else {
            if ((tx & 3) == 0) {
                for (int i = 0; i < 4; i++) {
                    if (tx + i < TILE_SIZE) {
                        As[ty][tx + i] = 0.0f;
                    }
                }
            }
        }

        // Load B tile into shared memory
        if (bRowStart + ty < N) {
            if ((tx & 3) == 0) {
                int globalCol = blockIdx.x * TILE_SIZE + tx;
                if (globalCol <= N - 4) {
                    const float4* B_vec = reinterpret_cast<const float4*>(B);
                    int index = (bRowStart + ty) * N + blockIdx.x * TILE_SIZE + tx;
                    float4 b_val = __ldg(&B_vec[index / 4]);
                    Bs[ty][tx]     = b_val.x;
                    Bs[ty][tx + 1] = b_val.y;
                    Bs[ty][tx + 2] = b_val.z;
                    Bs[ty][tx + 3] = b_val.w;
                } else {
                    for (int i = 0; i < 4; i++) {
                        int col_idx = blockIdx.x * TILE_SIZE + tx + i;
                        Bs[ty][tx + i] = (col_idx < N) ? __ldg(&B[(bRowStart + ty) * N + col_idx]) : 0.0f;
                    }
                }
            }
        } else {
            if ((tx & 3) == 0) {
                for (int i = 0; i < 4; i++) {
                    if (tx + i < TILE_SIZE) {
                        Bs[ty][tx + i] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        // Compute partial product for the tile
        if (row < N && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
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

    matmul_vectorized_aligned_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA) using vectorized __ldg() with 128-bit aligned accesses");
}
