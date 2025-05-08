#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Threshold below which a single thread computes the summation sequentially
#define REDUCTION_THRESHOLD 32

// Each block computes one output element C[row, col] for a lower triangular matrix multiplication
// where C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]

__global__ void parallel_reduction_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Flattened block index corresponds to one valid (row, col) pair with row >= col.
    int idx = blockIdx.x;

    // Compute the row index from the flattened index using the quadratic formula:
    // row = floor((sqrt(8*idx + 1) - 1) / 2)
    float idx_f = (float)idx;
    int row = (int)floorf((sqrtf(8.0f * idx_f + 1.0f) - 1.0f) * 0.5f);
    int base = row * (row + 1) / 2;  // number of elements before this row
    int col = idx - base;  // since each row has (row+1) valid columns, col in [0, row]
    
    // Now, for this (row, col), the summation runs from k = col to k = row.
    int terms = row - col + 1;  

    // If the number of terms is small, let only thread 0 compute the sum sequentially
    if (terms <= REDUCTION_THRESHOLD) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        return;
    }

    // For larger summation ranges, use parallel reduction with the threads in the block
    extern __shared__ float sdata[]; // shared memory for partial sums
    float partial = 0.0f;
    
    // Each thread computes a partial sum over a strided range
    for (int k = col + threadIdx.x; k <= row; k += blockDim.x) {
        partial += A[row * N + k] * B[k * N + col];
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();

    // Reduction in shared memory: assume blockDim.x is a power of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the result from the first thread of the block
    if (threadIdx.x == 0) {
        C[row * N + col] = sdata[0];
    }
}

// Host function exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // The output C is an N x N matrix
    auto C = torch::empty({N, N}, A.options());

    // Initialize the upper triangular part of C to 0 (for row < col)
    C.zero_();

    // Total number of lower triangular elements
    int total_elements = (N * (N + 1)) / 2;

    // Choose block size: using 128 threads per block for reduction
    int blockSize = 128;
    dim3 block(blockSize);
    dim3 grid(total_elements);

    // Launch the kernel with dynamic shared memory allocation
    size_t sharedMemSize = blockSize * sizeof(float);
    parallel_reduction_kernel<<<grid, block, sharedMemSize>>>(
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
    m.def("forward", &forward, "Parallel reduction triangular matmul (CUDA)");
}
