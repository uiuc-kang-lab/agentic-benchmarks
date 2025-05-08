#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimal_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const bool use_vectorized
) {
    int row = blockIdx.x;
    
    if (row < N) {
        float a_val = A[row];

        if (use_vectorized) {
            const float4* B_vec = reinterpret_cast<const float4*>(B);
            float4* C_vec = reinterpret_cast<float4*>(C);
            
            const int vec_total = M / 4;
            const int tid = threadIdx.x;
            const int total_threads = blockDim.x;
            
            for (int idx = tid; idx < vec_total; idx += total_threads) {
                float4 b_val = B_vec[row * vec_total + idx];
                float4 c_val;
                c_val.x = a_val * b_val.x;
                c_val.y = a_val * b_val.y;
                c_val.z = a_val * b_val.z;
                c_val.w = a_val * b_val.w;

                C_vec[row * vec_total + idx] = c_val;
            }
        } else {
            int tid = threadIdx.x;
            const int total_threads = blockDim.x;
            
            int main_end = (M / total_threads) * total_threads;
            
            for (int col = tid; col < main_end; col += total_threads) {
                int idx = row * M + col;
                C[idx] = a_val * B[idx];
            }

            for (int col = main_end + tid; col < M; col += total_threads) {
                int idx = row * M + col;
                C[idx] = a_val * B[idx];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    bool use_vectorized = (M >= 512) && (M % 4 == 0);
    
    // Experiment with different block sizes
    const int experiment_block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256;  // Default choice based on previous experiments
    //Here, speed measurement could select the block optimal size.
    
    for (int block_size : experiment_block_sizes) {
        int threads = (M > block_size) ? block_size : (((M + block_size - 1) / block_size) * block_size);
        
        if (use_vectorized) {
            const int blocks = (N * M + threads * 4 - 1) / (threads * 4);
            optimal_diag_matmul_kernel<<<blocks, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                N, M, true);
        } else {
            dim3 grid(N);
            optimal_diag_matmul_kernel<<<grid, threads>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                N, M, false);
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimal block diagonal matrix multiplication");
}
