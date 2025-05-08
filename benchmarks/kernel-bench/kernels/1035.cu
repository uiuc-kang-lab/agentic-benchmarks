#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128  // 128 threads per block (4 warps per block)

// Device function using inline to access matrix elements with optional transpose
__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel: Each warp computes one element of C using warp-level primitives
__global__ void warp_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc,
                                    bool transA, bool transB) {
    // Compute global thread id and warp id
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / WARP_SIZE;  // each warp consists of 32 threads
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Each warp computes one output element of C
    int total_elements = M * N;
    if (warp_id < total_elements) {
        int i = warp_id / N;  // row index in C
        int j = warp_id % N;  // column index in C

        float sum = 0.0f;
        // Each lane in the warp processes part of the K-dimension
        for (int k = lane; k < K; k += WARP_SIZE) {
            float a_val = get_element(A, i, k, lda, transA);
            float b_val = get_element(B, k, j, ldb, transB);
            sum += a_val * b_val;
        }
        
        // Warp-level reduction using __shfl_down_sync to sum partial sums
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Lane 0 writes the complete result for C[i,j]
        if (lane == 0) {
            C[i * ldc + j] = sum;
        }
    }
}

// Host function: Set up parameters and launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices.");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices.");
    }

    // Determine dimensions and transpose flags
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is (M x K), B is (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is (K x M), so it's stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is (N x K), stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are transposed
        transA = true;
        transB = true;
        M = A_cols;
        K = A_rows;
        N = B_rows;
        lda = A.stride(1);
        ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    ldc = N;
    auto C = torch::empty({M, N}, A.options());

    // Each warp computes one element of C => total warps required
    int total_elements = M * N;
    // Each block has BLOCK_SIZE threads (i.e., BLOCK_SIZE/WARP_SIZE warps per block)
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int blocks = (total_elements + warps_per_block - 1) / warps_per_block;

    warp_matmul_kernel<<<blocks, BLOCK_SIZE>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB
    );
    
    cudaDeviceSynchronize();
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-level optimized tall-and-skinny matrix multiplication (CUDA)");
}
