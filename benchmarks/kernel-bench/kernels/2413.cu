#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one element of C per block by reducing partial dot products computed by threads in the block.
// Each thread accumulates a partial sum over a strided section of the K dimension, stores it in shared memory,
// and then a two-phase reduction is performed: a binary tree reduction in shared memory followed by a final warp-level reduction
// using __shfl_down_sync, ensuring minimal synchronization overhead for the final stages.

__global__ void matmul_shared_warp_reduction_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int M, int N, int K) {
    // Each block computes one element C[i, j]
    int i = blockIdx.y;  // row index of C
    int j = blockIdx.x;  // column index of C
    int tid = threadIdx.x; // thread index within the block

    float partial_sum = 0.0f;
    // Each thread processes a portion of the K dimension using striding
    for (int k = tid; k < K; k += blockDim.x) {
        // B is stored in non-transposed form but used as transposed: C[i,j] = dot(A[i,:], B[j,:])
        partial_sum += A[i * K + k] * B[k * N + j];
    }

    // Allocate shared memory for reduction
    extern __shared__ float sdata[];
    sdata[tid] = partial_sum;
    __syncthreads();

    // Intra-block reduction using shared memory (binary tree reduction)
    // Reduce until we have 32 or fewer elements
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final reduction using warp-level primitives (no need for __syncthreads in a warp)
    float sum = (tid < 32) ? sdata[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write the final result from thread 0 of the block
    if (tid == 0) {
        C[i * N + j] = sum;
    }
}

// Forward function callable from PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    // Create output tensor C of size (M x N)
    auto C = torch::empty({M, N}, A.options());
    
    // Configure kernel launch: one block per output element
    // Block size: use 256 threads (can be tuned) and shared memory of 256 * sizeof(float)
    const int threads = 256;
    dim3 grid(N, M);
    size_t sharedMemSize = threads * sizeof(float);

    matmul_shared_warp_reduction_kernel<<<grid, threads, sharedMemSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using shared memory and warp-level reduction (CUDA)");
}
