#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void block_optimized_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int num_tiles
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[TILE_SIZE * TILE_SIZE];

    // Convert 1D block index to 2D coordinates
    int by = blockIdx.x / num_tiles;
    int bx = blockIdx.x % num_tiles;
    
    // Only process blocks in lower triangle
    if (by < bx) return;

    // Thread coordinates within the block
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;

    // Global coordinates
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (by - bx + 1) && (bx + t) * TILE_SIZE < N; ++t) {
        // Load tile from A
        if (row < N && (bx + t) * TILE_SIZE + tx < N) {
            As[ty * TILE_SIZE + tx] = __ldg(&A[row * N + (bx + t) * TILE_SIZE + tx]);
        } else {
            As[ty * TILE_SIZE + tx] = 0.0f;
        }

        // Load tile from B
        if ((bx + t) * TILE_SIZE + ty < N && col < N) {
            Bs[ty * TILE_SIZE + tx] = __ldg(&B[((bx + t) * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty * TILE_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty * TILE_SIZE + k] * Bs[k * TILE_SIZE + tx];
        }

        __syncthreads();
    }

    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::zeros_like(A);

    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    // Only launch blocks for lower triangle
    const int num_blocks = (num_tiles * (num_tiles + 1)) / 2;
    
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    const int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    block_optimized_triangular_mm_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        num_tiles
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block optimized triangular matrix multiplication (CUDA)");
}