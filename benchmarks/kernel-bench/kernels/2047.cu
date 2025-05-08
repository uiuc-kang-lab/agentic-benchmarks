#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define UNROLL_FACTOR 4

// Balanced kernel using a full rectangular grid to distribute work evenly among thread blocks.
// Each thread computes one element of C. If the thread corresponds to an element above the diagonal (row < col), it writes zero.
// For valid lower-triangular elements (row >= col), the dot product is computed from k = col to k = row using loop unrolling and vectorized loads for A.
// __ldg() is used to fetch B elements through the read-only cache.

__global__ void balanced_fullgrid_triangular_mm_kernel(const float* __restrict__ A,
                                                         const float* __restrict__ B,
                                                         float* __restrict__ C,
                                                         int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    // Only compute for lower-triangular region
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int start = col;
    int end = row + 1;  // inclusive upper bound
    int len = end - start;
    int unroll = (len / UNROLL_FACTOR) * UNROLL_FACTOR;

    // Loop unrolling with vectorized loads from A (which is row-major and thus contiguous)
    for (int k = start; k < start + unroll; k += UNROLL_FACTOR) {
        // Assume A is properly aligned for float4 loads
        float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
        float b0 = __ldg(&B[k * N + col]);
        float b1 = __ldg(&B[(k + 1) * N + col]);
        float b2 = __ldg(&B[(k + 2) * N + col]);
        float b3 = __ldg(&B[(k + 3) * N + col]);
        sum += a_vec.x * b0 + a_vec.y * b1 + a_vec.z * b2 + a_vec.w * b3;
    }

    // Process any remaining iterations
    for (int k = start + unroll; k < end; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    balanced_fullgrid_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced full-grid triangular matrix multiplication (CUDA)");
}
