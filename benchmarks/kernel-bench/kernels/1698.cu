#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define the warp size for warp-level primitives
#define WARP_SIZE 32

// Kernel: each block computes one lower-triangular element C(i,j) = sum_{k=j}^{i} A(i,k)*B(k,j)
// The lower-triangular elements are packed into a linear index (idx) over [0, N*(N+1)/2).
// Within each block, threads collaboratively compute the reduction over k using both shared memory and warp-level intrinsics (__shfl_down_sync).
__global__ void triangular_mm_coop_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N,
                                           int total_elements) {
    // Each block is assigned one output element in the lower triangle.
    int idx = blockIdx.x;
    if (idx >= total_elements) return;

    // Map linear index idx to matrix coordinates (i, j) for lower triangular part
    // Solve: i such that i*(i+1)/2 <= idx < (i+1)*(i+2)/2
    int i = (int)floor((sqrtf(8.0f * idx + 1.0f) - 1.0f) * 0.5f);
    int base = i * (i + 1) / 2;
    int j = idx - base;
    if (i >= N || j > i) return;

    // Each block computes the sum for C(i,j) = sum_{k=j}^{i} A(i,k) * B(k,j)
    float sum = 0.0f;
    // Distribute the summation over threads in the block
    for (int k = j + threadIdx.x; k <= i; k += blockDim.x) {
        sum += A[i * N + k] * B[k * N + j];
    }

    // Intra-warp reduction using warp-level shuffle
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Use shared memory to combine results from different warps in the block
    extern __shared__ float sdata[]; // size: (blockDim.x / WARP_SIZE) floats
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    __syncthreads();

    // Now, have the first few threads (one per warp) perform final reduction
    int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < numWarps) {
        float warpSum = sdata[threadIdx.x];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (threadIdx.x == 0) {
            C[i * N + j] = warpSum;
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Initialize output tensor C with zeros. This takes care of the upper triangular region.
    auto C = torch::zeros_like(A);

    // Total number of lower triangular elements
    int total_elements = N * (N + 1) / 2;

    // Use a fixed number of threads per block for reduction; e.g., 128 threads
    int blockSize = 128;
    dim3 threads(blockSize);
    // Launch one block per lower triangular element
    dim3 blocks(total_elements);

    // Shared memory size: one float per warp in the block
    int numWarps = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
    int sharedMemSize = numWarps * sizeof(float);

    triangular_mm_coop_kernel<<<blocks, threads, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        total_elements
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular Matrix Multiplication with Shared Memory Reduction (CUDA)");
}
