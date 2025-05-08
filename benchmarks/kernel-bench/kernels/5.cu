#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4  // Using float4 for vectorized loads

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_coalesced_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Convert to vectorized indices
    int vec_tx = tx * VECTOR_SIZE;
    
    // Global row and column
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum[VECTOR_SIZE] = {0.0f};

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load A tile using vectorized loads when possible
        if (row < N && m * TILE_SIZE + vec_tx < N) {
            float4* As_vec = (float4*)&As[ty][vec_tx];
            float4* A_vec = (float4*)&A[row * N + m * TILE_SIZE + vec_tx];
            if (vec_tx + VECTOR_SIZE <= TILE_SIZE) {
                *As_vec = *A_vec;
            } else {
                for (int i = 0; i < min(VECTOR_SIZE, TILE_SIZE - vec_tx); ++i) {
                    As[ty][vec_tx + i] = A[row * N + m * TILE_SIZE + vec_tx + i];
                }
            }
        }

        // Load B tile using coalesced access pattern
        if (m * TILE_SIZE + ty < N && col < N) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        }

        __syncthreads();

        // Compute partial products with vectorized computation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float bval = Bs[k][tx];
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                if (vec_tx + v < TILE_SIZE) {
                    sum[v] += As[ty][k] * bval;
                }
            }
        }

        __syncthreads();
    }

    // Store results using coalesced writes
    if (row < N && col < N) {
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; ++v) {
            if (vec_tx + v < TILE_SIZE) {
                C[row * N + col] = sum[v];
            }
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

    dim3 threadsPerBlock(TILE_SIZE/VECTOR_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_coalesced_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel (CUDA)");
}