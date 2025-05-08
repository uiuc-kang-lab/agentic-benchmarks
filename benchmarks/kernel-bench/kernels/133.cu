#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Function to calculate the best block size based on matrix sizes
int calculate_best_block_size(int M, int N, int K) {
    int size = max(M, max(N, K));
    if (size <= 64) return 64;
    else if (size <= 128) return 128;
    else if (size <= 256) return 256;
    else return 512;
}

// Static cuBLAS handle to avoid recreation overhead
static cublasHandle_t handle = nullptr;

// Dynamic blocksize matrix multiplication kernel
__global__ void dynamic_blocksize_matmul_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int M, int N, int K, int block_size) {
    extern __shared__ float shared_data[];
    float* As = shared_data;
    float* Bs = shared_data + block_size * block_size;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * block_size + ty;
    int col = bx * block_size + tx;
    float sum = 0.0f;

    int numTiles = (K + block_size - 1) / block_size;
    for (int t = 0; t < numTiles; ++t) {
        if (row < M && t * block_size + tx < K) {
            As[ty * block_size + tx] = A[row * K + t * block_size + tx];
        } else {
            As[ty * block_size + tx] = 0.0f;
        }

        if (t * block_size + ty < K && col < N) {
            Bs[ty * block_size + tx] = B[(t * block_size + ty) * N + col];
        } else {
            Bs[ty * block_size + tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < block_size; ++k) {
            sum += As[ty * block_size + k] * Bs[k * block_size + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A,
                          const torch::Tensor &B,
                          torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Calculate the optimal block size
    int block_size = calculate_best_block_size(M, N, K);

    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
        size_t sharedMemSize = 2 * block_size * block_size * sizeof(float);
        dynamic_blocksize_matmul_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, M, N, K, block_size);
    } else {
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device()).requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Dynamic block size matrix multiplication (CUDA)");
}
