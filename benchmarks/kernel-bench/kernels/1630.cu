#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Kernel: Each block computes one element C[row, col] for the upper triangular matrix multiplication.
// The dot product for each element C[row, col] = sum_{k=row}^{col} A[row,k] * B[k,col] is computed in parallel
// by splitting the summation across threads in the block. Partial sums are stored in shared memory,
// then reduced through intra-block reduction and final warp-level reduction using __shfl_down_sync().

__global__ void upper_triangular_matmul_parallel_kernel(const float* A, const float* B, float* C, int N) {
    // Map each block to a specific (row, col) of C
    int row = blockIdx.y;  // blockIdx.y maps to row
    int col = blockIdx.x;  // blockIdx.x maps to col

    if (row < N && col < N && row <= col) {
        // Range for summation: k from row to col (inclusive)
        int len = col - row + 1;  // number of multiplications required
        float partial_sum = 0.0f;

        // Each thread computes a partial sum over a subset of indices in the range [row, col]
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            int k = row + i;
            partial_sum += A[row * N + k] * B[k + col * N];
        }

        // Use shared memory to store partial sums from all threads in the block
        extern __shared__ float sdata[]; // dynamically allocated shared memory
        sdata[threadIdx.x] = partial_sum;
        __syncthreads();

        // Intra-block reduction using shared memory. Assume blockDim.x is a power of two.
        // Reduce until we have at most 32 elements.
        for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
            if (threadIdx.x < stride) {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Final warp-level reduction for the last 32 elements using __shfl_down_sync
        if (threadIdx.x < 32) {
            float val = sdata[threadIdx.x];
            // Unroll warp-level reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (threadIdx.x == 0) {
                C[row * N + col] = val;
            }
        }
    }
}

// Host function that launches the kernel.
// Each block is assigned one element of the output matrix (for valid upper triangular indices).
// The grid is set to (N, N) and threads within a block perform a parallel reduction on the dot product.
// The third parameter for shared memory allocation is set to blockDim.x * sizeof(float).

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Launch configuration: grid covers (col, row) and each block computes one C[row,col]
    dim3 gridDim(N, N);
    int threads = 128;  
    dim3 blockDim(threads);

    // Launch the kernel with dynamically allocated shared memory of size (threads * sizeof(float))
    upper_triangular_matmul_parallel_kernel<<<gridDim, blockDim, threads * sizeof(float)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matmul with shared memory and warp-level reduction");
}
