#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;
const int THREAD_STRIDE = 4;

__global__ void matmul_transposed_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Base indices for strided computation
    int m_base = by * (TILE_SIZE * THREAD_STRIDE) + ty;
    int n_base = bx * (TILE_SIZE * THREAD_STRIDE) + tx;

    float c_vals[THREAD_STRIDE][THREAD_STRIDE] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Load tiles with stride
        for (int i = 0; i < THREAD_STRIDE; i++) {
            for (int j = 0; j < THREAD_STRIDE; j++) {
                int m = m_base + i * TILE_SIZE;
                if (m < M && (k_offset + tx) < K) {
                    As[ty + i * TILE_SIZE/THREAD_STRIDE][tx] = A[m * K + k_offset + tx];
                } else {
                    As[ty + i * TILE_SIZE/THREAD_STRIDE][tx] = 0.0f;
                }

                int n = n_base + j * TILE_SIZE;
                if (n < N && (k_offset + ty) < K) {
                    Bs[ty][tx + j * TILE_SIZE/THREAD_STRIDE] = B[n * K + k_offset + ty];
                } else {
                    Bs[ty][tx + j * TILE_SIZE/THREAD_STRIDE] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute with stride
        for (int i = 0; i < THREAD_STRIDE; i++) {
            for (int j = 0; j < THREAD_STRIDE; j++) {
                for (int k = 0; k < TILE_SIZE; ++k) {
                    c_vals[i][j] += As[ty + i * TILE_SIZE/THREAD_STRIDE][k] * 
                                    Bs[k][tx + j * TILE_SIZE/THREAD_STRIDE];
                }
            }
        }

        __syncthreads();
    }

    // Store results with stride
    for (int i = 0; i < THREAD_STRIDE; i++) {
        for (int j = 0; j < THREAD_STRIDE; j++) {
            int m = m_base + i * TILE_SIZE;
            int n = n_base + j * TILE_SIZE;
            if (m < M && n < N) {
                C[m * N + n] = c_vals[i][j];
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

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE * THREAD_STRIDE - 1) / (TILE_SIZE * THREAD_STRIDE),
              (M + TILE_SIZE * THREAD_STRIDE - 1) / (TILE_SIZE * THREAD_STRIDE));
    dim3 block(TILE_SIZE/THREAD_STRIDE, TILE_SIZE/THREAD_STRIDE);
    
    matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}