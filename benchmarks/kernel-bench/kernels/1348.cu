#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void minimal_sync_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    constexpr int ROWS_PER_BLOCK = 4;
    __shared__ float shared_diag[ROWS_PER_BLOCK];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    
    const int rows_this_block = min(ROWS_PER_BLOCK, static_cast<int>(N - block_row_start));
    
    if (tid < rows_this_block) {
        shared_diag[tid] = A[block_row_start + tid];
    }
    __syncthreads();
    
    if (M % 4 == 0) {
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);
        const int vec_M = M >> 2;
        
        for (int row_offset = 0; row_offset < rows_this_block; row_offset++) {
            const int row = block_row_start + row_offset;
            const float a_val = shared_diag[row_offset];
            const int row_vec_offset = row * vec_M;
            
            for (int j = tid; j < vec_M; j += num_threads) {
                const float4 b_val = B_vec[row_vec_offset + j];
                float4 c_val;
                c_val.x = a_val * b_val.x;
                c_val.y = a_val * b_val.y;
                c_val.z = a_val * b_val.z;
                c_val.w = a_val * b_val.w;
                C_vec[row_vec_offset + j] = c_val;
            }
        }
    } else {
        for (int row_offset = 0; row_offset < rows_this_block; row_offset++) {
            const int row = block_row_start + row_offset;
            const float a_val = shared_diag[row_offset];
            const int row_offset_M = row * M;
            
            for (int j = tid; j < M; j += num_threads) {
                C[row_offset_M + j] = a_val * B[row_offset_M + j];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    constexpr int ROWS_PER_BLOCK = 4;
    const int threads = 256;
    const int blocks = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    minimal_sync_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Minimal sync diagonal matrix multiplication");
}