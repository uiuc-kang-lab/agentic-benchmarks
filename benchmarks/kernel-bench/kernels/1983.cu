#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes C = tril(A * B) for lower triangular matrices A and B.
// Each CUDA block is responsible for one output element C[i,j] (with i >= j).
// Within the block, multiple threads cooperatively compute the dot product sum_{k=j}^{i} A[i,k] * B[k,j] by
// performing a grid-stride loop over the summation range, followed by an intra-block reduction in shared memory.
// Since each block exclusively computes one output element, no global atomic operations are used.

__global__ void triangular_mm_kernel_cooperative(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {

    // Map each block to an output element (i, j)
    int i = blockIdx.y; // row index
    int j = blockIdx.x; // column index

    // Check bounds
    if (i >= N || j >= N) return;

    // For upper triangular part, output is zero
    if (i < j) {
        if (threadIdx.x == 0)
            C[i * N + j] = 0.0f;
        return;
    }

    // Compute the dot product for C[i, j] = sum_{k=j}^{i} A[i,k] * B[k,j]
    float sum = 0.0f;
    int tid = threadIdx.x;
    // Each thread processes a disjoint chunk of the summation index [j, i]
    for (int k = j + tid; k <= i; k += blockDim.x) {
        sum += A[i * N + k] * B[k * N + j];
    }

    // Intra-block reduction using shared memory
    extern __shared__ float sdata[]; // size = blockDim.x * sizeof(float)
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within the block (assume blockDim.x is a power of 2)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the final result
    if (tid == 0) {
        C[i * N + j] = sdata[0];
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
    auto C = torch::empty_like(A);

    // Launch a 2D grid where each block computes one output element C[i,j]
    // Grid dimensions: x-dim for columns, y-dim for rows
    dim3 grid(N, N);
    // Use a block of 128 threads; adjust as needed depending on N
    int threads = 128;
    size_t shared_mem = threads * sizeof(float);

    triangular_mm_kernel_cooperative<<<grid, threads, shared_mem>>>(
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
    m.def("forward", &forward, "Cooperative reduction triangular matrix multiplication (CUDA)");
}
