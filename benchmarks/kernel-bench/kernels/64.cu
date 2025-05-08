#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// We assign one warp (32 threads) to compute a single matrix element's dot product.
// Each warp's threads partition the k-loop, and a warp-level reduction with __shfl_down_sync() is used to sum partial products.
// Block configuration: blockDim.x = 32 (the warp lanes), blockDim.y = WARPS_PER_BLOCK.
// Grid configuration: gridDim.x = N (each block column computes one output column),
// and gridDim.y = ceil(N / WARPS_PER_BLOCK) so that each warp in a block computes one row element.

#define WARPS_PER_BLOCK 8
#define WARP_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void warp_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Determine thread's role within the warp and block
    int warp_lane = threadIdx.x;         // Lane ID within warp (0 ... 31)
    int warp_row = threadIdx.y;          // Which warp within the block

    // Compute the global row and col indices for the output matrix element computed by this warp
    int row = blockIdx.y * blockDim.y + warp_row;
    int col = blockIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    // Each warp partitions the k-loop among its 32 threads
    // Unroll the k-loop to improve throughput and use __ldg() for read-only loads
    int num_iters = (N - warp_lane + WARP_SIZE - 1) / WARP_SIZE;
    int i = 0;
    for (; i + 3 < num_iters; i += 4) {
        int k0 = warp_lane + i * WARP_SIZE;
        int k1 = warp_lane + (i + 1) * WARP_SIZE;
        int k2 = warp_lane + (i + 2) * WARP_SIZE;
        int k3 = warp_lane + (i + 3) * WARP_SIZE;
        float a0 = __ldg(&A[row * N + k0]);
        float b0 = __ldg(&B[k0 * N + col]);
        float a1 = __ldg(&A[row * N + k1]);
        float b1 = __ldg(&B[k1 * N + col]);
        float a2 = __ldg(&A[row * N + k2]);
        float b2 = __ldg(&B[k2 * N + col]);
        float a3 = __ldg(&A[row * N + k3]);
        float b3 = __ldg(&B[k3 * N + col]);
        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    for (; warp_lane + i * WARP_SIZE < N; i++) {
        int k = warp_lane + i * WARP_SIZE;
        sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
    }

    // Perform warp-level reduction to sum partial results
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Only lane 0 of each warp writes the final result
    if (warp_lane == 0) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Grid: one block per output column, with each block covering WARPS_PER_BLOCK rows
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    dim3 grid(N, (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    warp_matmul_kernel<<<grid, block>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized matrix multiplication kernel (CUDA)");
}
