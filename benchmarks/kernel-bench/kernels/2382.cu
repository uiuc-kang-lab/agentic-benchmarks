#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;
const int STRIDE_LOOPS = 4;

__global__ void matmul_transposed_strided_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[STRIDE_LOOPS][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int baseRow = by * TILE_SIZE * STRIDE_LOOPS;
    int baseCol = bx * TILE_SIZE;
    float c_val[STRIDE_LOOPS] = {0};

        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            int tk = t * TILE_SIZE;

            // Load As for multiple rows with stride
            if (tk + tx < K) {
                for (int i = 0; i < STRIDE_LOOPS; ++i) {
                    int row = m - ty + i * TILE_SIZE;
                    if (row < M) {
                        As[i][ty][tx] = A[row * K + tk + tx];
                    } else {
                        As[i][ty][tx] = 0.0f;
                    }
                }
            }

            // Load Bs
            if (n < N && tk + ty < K) {
                Bs[ty][tx] = B[n * K + tk + ty];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k) {
                #pragma unroll
                for (int i = 0; i < STRIDE_LOOPS; ++i) {
                    c_val[i] += As[i][ty][k] * Bs[k][tx];
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i = 0; i < STRIDE_LOOPS; ++i) {
            int out_row = m + i * TILE_SIZE;
            if (out_row < M && n < N) {
                atomicAdd(&C[out_row * N + n], c_val[i]);
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::zeros({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE * STRIDE_LOOPS - 1) / (TILE_SIZE * STRIDE_LOOPS));
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    matmul_transposed_strided_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (Strided CUDA)");
}
