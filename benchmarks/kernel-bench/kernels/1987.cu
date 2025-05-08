#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel: uses tuned block dimensions (32x16) to achieve both coalesced global memory accesses for B
// and higher occupancy from experiments. Thread.x indexes the col dimension to get coalesced loads from B.
__global__ void combined_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Calculate matrix indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    // Only compute for lower-triangular part; upper part is explicitly zero
    if (row < col) {
        C[row * N + col] = 0.0f;
    } else {
        float sum = 0.0f;
        // Note: start at k=col ensures memory accesses in B (B[k * N + col]) are coalesced across threads
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use block dimensions that combine optimal occupancy with memory coalescing.
    // 32 threads in x (matching warp size) ensure B accesses are coalesced,
    // while 16 threads in y increases SM utilization.
    const int bx = 32, by = 16;
    dim3 threads(bx, by);
    dim3 blocks((N + bx - 1) / bx, (N + by - 1) / by);

    combined_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Combined optimized triangular matmul (CUDA)");
}
