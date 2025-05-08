#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

__device__ float get_element(const float* matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void warp_tall_skinny_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        bool transA, bool transB) {
    // Calculate global C position
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= M * N) return;
    
    int row = warp_id / N;
    int col = warp_id % N;
    
    float acc = 0.0f;
    for (int k = lane_id; k < K; k += WARP_SIZE) {
        float a = get_element(A, row, k, lda, transA);
        float b = get_element(B, k, col, ldb, transB);
        acc += a * b;
    }
    
    // Warp reduction using shfl_down
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    
    if (lane_id == 0 && row < M && col < N) {
        C[row * ldc + col] = acc;
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Shape and stride calculation from reference implementation
    int64_t M, N, K, lda, ldb, ldc;
    bool transA, transB;
    
    M = A.size(0); N = B.size(1); K = A.size(1); lda = A.stride(0); ldb = B.stride(0); ldc = C.stride(0); transA = false; transB = false; // Assuming no transposition for simplicity
    
    auto C = torch::empty({M, N}, A.options());
    
    // Warp-based configuration
    int total_warps = M * N;
    int warps_per_block = 8;
    int block_size = warps_per_block * WARP_SIZE;
    dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);
    
    warp_tall_skinny_kernel<<<grid, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                                                 M, N, K, lda, ldb, ldc, transA, transB);
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tall_skinny_matmul_cuda, "Warp-optimized tall-skinny matmul (CUDA)");
}