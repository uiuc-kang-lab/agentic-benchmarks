#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel that computes C = A.T * B using shared memory and warp-level reduction.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
__global__ void sharedMemoryKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K,
                                   int M,
                                   int N) {
    // Shared memory for partial sums
    extern __shared__ float sharedSum[];

    // Calculate global row and column indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Each thread computes a partial sum
    float sum = 0.0f;
    for (int k = threadIdx.z; k < K; k += blockDim.z) {
        sum += A[k * M + i] * B[k * N + j];
    }

    // Store partial sum in shared memory
    int threadId = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    sharedSum[threadId] = sum;
    __syncthreads();

    // Reduce within the block using shared memory
    for (int stride = blockDim.z / 2; stride > 0; stride /= 2) {
        if (threadIdx.z < stride) {
            sharedSum[threadId] += sharedSum[threadId + stride * blockDim.x * blockDim.y];
        }
        __syncthreads();
    }

    // Final reduction using warp-level primitives
    if (threadIdx.z == 0) {
        float finalSum = sharedSum[threadIdx.y * blockDim.x + threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            finalSum += __shfl_down_sync(0xffffffff, finalSum, offset);
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            C[i * N + j] = finalSum;
        }
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, M) and B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes.
    const int THREADS_X = 16;
    const int THREADS_Y = 16;
    const int THREADS_Z = 4; // Use a small z-dimension for reduction
    dim3 blockDim(THREADS_X, THREADS_Y, THREADS_Z);
    dim3 gridDim((M + THREADS_X - 1) / THREADS_X, (N + THREADS_Y - 1) / THREADS_Y);

    // Calculate shared memory size
    size_t sharedMemSize = THREADS_X * THREADS_Y * THREADS_Z * sizeof(float);

    // Get raw pointers to the data.
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel.
    sharedMemoryKernel<<<gridDim, blockDim, sharedMemSize>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with shared memory and warp-level reduction (CUDA)");
}
