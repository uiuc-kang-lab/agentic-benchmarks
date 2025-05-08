#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__global__ void vectorized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    __shared__ float shared_diag[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    
    // Load diagonal element into shared memory
    if (tid < min((int64_t)BLOCK_SIZE, N - bid * BLOCK_SIZE)) {
        shared_diag[tid] = A[bid * BLOCK_SIZE + tid];
    }
    __syncthreads();
    
    // Calculate starting position for this thread block
    const int row_start = bid * BLOCK_SIZE;
    const int rows_this_block = min(BLOCK_SIZE, (int)(N - row_start));
    
    for (int row_offset = 0; row_offset < rows_this_block; row_offset++) {
        const int row = row_start + row_offset;
        const float diag_val = shared_diag[row_offset];
        
        // Process four elements at a time using float4
        int col = tid * 4;
        for (; col <= M - 4; col += num_threads * 4) {
            float4 b_vec = *reinterpret_cast<const float4*>(&B[row * M + col]);
            float4 c_vec;
            c_vec.x = diag_val * b_vec.x;
            c_vec.y = diag_val * b_vec.y;
            c_vec.z = diag_val * b_vec.z;
            c_vec.w = diag_val * b_vec.w;
            *reinterpret_cast<float4*>(&C[row * M + col]) = c_vec;
        }
        
        // Handle remaining elements
        for (int i = col + tid; i < M; i += num_threads) {
            C[row * M + i] = diag_val * B[row * M + i];
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int threads = BLOCK_SIZE;

    vectorized_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized diagonal matrix multiplication with shared memory");
}