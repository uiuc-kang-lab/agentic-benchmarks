#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define TILE_M 8
#define TILE_N 4

__global__ void matmul_hierarchical_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    int warp_row = warp_id / TILE_N;
    int warp_col = warp_id % TILE_N;

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    int m = block_row + warp_row;  // Row index of warp within block
    int n = block_col + warp_col;  // Column index of warp within block

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        if ((block_row + threadIdx.y) < M && (k_offset + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[(block_row + threadIdx.y) * K + k_offset + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if ((block_col + threadIdx.y) < N && (k_offset + threadIdx.x) < K) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(block_col + threadIdx.y) * K + k_offset + threadIdx.x]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = lane; k < TILE_SIZE; k += WARP_SIZE) {
            sum += As[warp_row * TILE_M + threadIdx.y][k] * Bs[k][warp_col * TILE_N + threadIdx.x];
        }

        __syncthreads();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0 && m < M && n < N) {
        C[m * N + n] = sum;
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_hierarchical_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication optimized with hierarchical tiling and warp-level reduction (CUDA)");
}
