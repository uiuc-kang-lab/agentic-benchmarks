#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel assigns one CUDA block per lower-triangular output element C[i,j] (with i >= j).
// Within each block, the summation for C[i,j] = sum_{k=j}^{i} A[i,k] * B[k,j] is distributed over threads,
// and a shared-memory reduction is performed to compute the final result. There is no use of global
// atomic operations because each output element is computed exclusively by one block, thereby
// minimizing global contention while ensuring correctness.

__global__ void triangular_mm_kernel_parallel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    // Each block is responsible for one lower triangular element.
    // The total number of lower triangular elements is N*(N+1)/2.
    int idx = blockIdx.x;

    // Recover (i, j) from the linear index idx using the formula:
    // idx = i*(i+1)/2 + j, with 0 <= j <= i. Solve for i:
    int i = (int)floorf((sqrtf(8.0f * idx + 1.0f) - 1.0f) * 0.5f);
    int j = idx - (i * (i + 1)) / 2;

    if (i >= N || j > i) return;  // Safety check (should not occur if grid is set correctly)

    float sum = 0.0f;
    // Distribute the summation over k from j to i among the threads in the block
    for (int k = j + threadIdx.x; k <= i; k += blockDim.x) {
        sum += A[i * N + k] * B[k * N + j];
    }

    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory assuming blockDim.x is a power of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the result for this element
    if (threadIdx.x == 0) {
        C[i * N + j] = sdata[0];
    }
}

// C++ interface exposed to PyTorch via pybind11
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

    // Total number of lower-triangular elements
    int numElements = (N * (N + 1)) / 2;

    // Choose a thread block size for the reduction along the k-dimension
    int threads = 256;
    dim3 blocks(numElements);

    // Launch the kernel with one block per lower-triangular element and allocate shared memory
    triangular_mm_kernel_parallel<<<blocks, threads, threads * sizeof(float)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Parallel lower triangular matrix multiplication (CUDA)");
}
