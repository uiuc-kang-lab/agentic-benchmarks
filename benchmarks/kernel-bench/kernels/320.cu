#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define PADDING 2

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + PADDING];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Remap threads for coalesced memory access
    const int tid = ty * TILE_SIZE + tx;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Calculate global indices with stride for coalesced access
    const int row = by * TILE_SIZE + warp_id;
    const int col = bx * TILE_SIZE + lane_id;
    
    float sum = 0.0f;
    
    // Base pointers for current batch
    const float* batch_A = A + bz * M * K;
    const float* batch_B = B + bz * K * N;
    
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        // Collaborative loading with coalesced access pattern
        if (tid < TILE_SIZE) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i += 4) {
                if ((row < M) && (tile + i + lane_id/8) < K) {
                    int idx = row * K + tile + i + lane_id/8;
                    As[tid][lane_id/8] = batch_A[idx];
                }
                if ((tile + tid) < K && (col + i) < N) {
                    int idx = (tile + tid) * N + col + i;
                    Bs[tid][lane_id/8] = batch_B[idx];
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(As[ty][k], Bs[k][tx], sum);
        }
        
        __syncthreads();
    }
    
    // Store result with coalesced writes
    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA)");
}