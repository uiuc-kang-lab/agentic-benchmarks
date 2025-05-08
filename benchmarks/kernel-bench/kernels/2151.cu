#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_WARP 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE+1]; // +1 for bank conflict avoidance
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE+1];

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;

    // Warp-aligned coordinates
    const int row = block_row + warp_id;
    const int col = block_col + lane_id;
    
    // Entire warp either processes lower triangle or zeros upper
    const bool warp_in_lower = (block_row + TILE_SIZE > block_col);
    
    if (!warp_in_lower) {
        if (row < N && col < N && col > row) {
            C[row*N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;

    // Calculate valid k range for this warp
    const int k_start = max(block_col, col);
    const int k_end = min(block_row + TILE_SIZE, row + 1);

    for (int t = k_start/TILE_SIZE; t <= k_end/TILE_SIZE; ++t) {
        // Collaborative loading with warp-aligned access
        const int load_row = block_row + warp_id;
        const int load_col = t*TILE_SIZE + lane_id;
        if (load_row < N && load_col <= row && load_col < N) {
            shared_A[warp_id][lane_id] = A[load_row*N + load_col];
        } else {
            shared_A[warp_id][lane_id] = 0.0f;
        }

        const int b_load_row = t*TILE_SIZE + warp_id;
        const int b_load_col = block_col + lane_id;
        if (b_load_row >= col && b_load_row < N && b_load_col < N) {
            shared_B[warp_id][lane_id] = B[b_load_row*N + b_load_col];
        } else {
            shared_B[warp_id][lane_id] = 0.0f;
        }

        __syncwarp();

        // Compute with unrolled inner loop
        #pragma unroll
        for (int k_sub = 0; k_sub < TILE_SIZE; ++k_sub) {
            const int k = t*TILE_SIZE + k_sub;
            if (k >= k_start && k <= k_end) {
                sum += shared_A[warp_id][k_sub] * shared_B[k_sub][lane_id];
            }
        }
    }

    if (row < N && col < N && col <= row) {
        C[row*N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrix size mismatch");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(THREADS_PER_WARP, WARPS_PER_BLOCK);
    dim3 blocks((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication");
}