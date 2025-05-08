#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the stride for how many columns each warp processes
const int COLUMN_STRIDE = 4;

__global__ void warp_matmul_strided_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    // Each warp processes COLUMN_STRIDE columns
    const unsigned int lane = threadIdx.x;     // lane id [0,31]
    const unsigned int warpId = threadIdx.y;   // warp id within block
    const int warpSize = 32;
    
    // Calculate base row and column indices
    const int row = blockIdx.y * blockDim.y + warpId;
    const int baseCol = blockIdx.x * COLUMN_STRIDE;
    
    // Each warp processes multiple columns with stride
    if (row < M) {
        // Pre-load row of A into registers to reuse across columns
        const int num_chunks = (K + warpSize - 1) / warpSize;
    const int MAX_K_CHUNKS = 128;
    float a_vals[MAX_K_CHUNKS];
    for (int i = 0; i < num_chunks; i++) { a_vals[i] = 0.0f; }  // preinitialize A values
        
        #pragma unroll
        for (int k_chunk = 0; k_chunk < (K + warpSize - 1) / warpSize; k_chunk++) {
            const int k = k_chunk * warpSize + lane;
            if (k < K) {
                a_vals[k_chunk] = __ldg(&A[row * K + k]);
            }
        }
        
        // Process COLUMN_STRIDE columns per warp
        #pragma unroll
        for (int col_offset = 0; col_offset < COLUMN_STRIDE; col_offset++) {
            const int col = baseCol + col_offset;
            if (col < N) {
                float sum = 0.0f;
                
                // Compute partial sums using pre-loaded A values
                #pragma unroll
                for (int k_chunk = 0; k_chunk < (K + warpSize - 1) / warpSize; k_chunk++) {
                    const int k = k_chunk * warpSize + lane;
                    if (k < K) {
                        sum += a_vals[k_chunk] * __ldg(&B[col * K + k]);
                    }
                }
                
                // Warp-level reduction
                unsigned int mask = 0xffffffff;
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(mask, sum, offset);
                }
                
                // First thread in warp writes result
                if (lane == 0) {
                    C[row * N + col] = sum;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Configure launch parameters
    const int warpSize = 32;
    const int warpsPerBlock = 8;
    dim3 block(warpSize, warpsPerBlock);
    
    // Grid dimensions adjusted for strided processing
    dim3 grid((N + COLUMN_STRIDE - 1) / COLUMN_STRIDE, (M + warpsPerBlock - 1) / warpsPerBlock);

    warp_matmul_strided_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided warp-level matrix multiplication with transposed B (CUDA)");
}