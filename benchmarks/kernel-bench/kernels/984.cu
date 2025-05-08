#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

// Constant memory for configuration parameters
__constant__ int const_dims[6];  // M, N, K, lda, ldb, ldc
__constant__ bool const_trans[2];  // transA, transB

// Helper to fetch matrix elements considering transpose
__device__ inline float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel with manual loop unrolling for critical loops
__global__ void unrolled_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
    // Load configuration from constant memory
    const int M = const_dims[0];
    const int N = const_dims[1];
    const int K = const_dims[2];
    const int lda = const_dims[3];
    const int ldb = const_dims[4];
    const int ldc = const_dims[5];
    const bool transA = const_trans[0];
    const bool transB = const_trans[1];

    // Calculate block's starting indices
    int block_row = blockIdx.y * (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    int block_col = blockIdx.x * BLOCK_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Shared memory tiles
    __shared__ float As[ELEMENTS_PER_THREAD][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Accumulators: each thread computes ELEMENTS_PER_THREAD output elements
    float C_values[ELEMENTS_PER_THREAD] = {0.0f};

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int tiledK = t * BLOCK_SIZE;

        // Load tile of B into shared memory with bounds check
        if (tiledK + thread_row < K && block_col + thread_col < N)
            Bs[thread_row][thread_col] = get_element(B, tiledK + thread_row, block_col + thread_col, ldb, transB);
        else
            Bs[thread_row][thread_col] = 0.0f;

        // Load a tile of A into shared memory. Each thread loads ELEMENTS_PER_THREAD elements
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            int row = block_row + e * BLOCK_SIZE + thread_row;
            if (row < M && tiledK + thread_col < K)
                As[e][thread_row][thread_col] = get_element(A, row, tiledK + thread_col, lda, transA);
            else
                As[e][thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        // Multiply the loaded tiles
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                C_values[e] += As[e][thread_row][k] * Bs[k][thread_col];
            }
        }

        __syncthreads();
    }

    // Write the computed results back to global memory
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
        int row = block_row + e * BLOCK_SIZE + thread_row;
        int col = block_col + thread_col;
        if (row < M && col < N) {
            C[row * ldc + col] = C_values[e];
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    int dims[6];
    dims[0] = A.size(0);   // M
    dims[1] = B.size(1);   // N
    dims[2] = A.size(1);   // K
    dims[3] = A.stride(0); // lda
    dims[4] = B.stride(0); // ldb
    dims[5] = B.size(1);   // ldc

    bool trans[2] = {false, false};

    // Copy configuration to constant memory
    cudaMemcpyToSymbol(const_dims, dims, sizeof(dims));
    cudaMemcpyToSymbol(const_trans, trans, sizeof(trans));

    auto C = torch::empty({dims[0], dims[1]}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dims[1] + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (dims[0] + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));

    unrolled_matmul_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with unrolled loops optimization (CUDA)");
}
