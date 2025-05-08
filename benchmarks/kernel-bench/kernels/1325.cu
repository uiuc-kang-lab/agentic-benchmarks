#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hybrid_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const bool use_vectorized
) {
    if (use_vectorized) {
        // Vectorized approach for large matrices where M is divisible by 4
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        const int total = N * M;
        const int vec_total = total / 4;
        
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);
        
        for (int idx = tid; idx < vec_total; idx += stride) {
            const int base_idx = idx * 4;
            const int row = base_idx / M;
            const float a_val = A[row];
            
            float4 b_val = B_vec[idx];
            float4 c_val;
            c_val.x = a_val * b_val.x;
            c_val.y = a_val * b_val.y;
            c_val.z = a_val * b_val.z;
            c_val.w = a_val * b_val.w;
            
            C_vec[idx] = c_val;
        }
    } else {
        // Row-based approach for smaller matrices or when M is not divisible by 4
        int row = blockIdx.x;
        if (row < N) {
            float a_val = A[row];
            const int main_end = (M / blockDim.x) * blockDim.x;
            
            // Main loop with coalesced access
            for (int j = threadIdx.x; j < main_end; j += blockDim.x) {
                int idx = row * M + j;
                C[idx] = a_val * B[idx];
            }
            
            // Handle remaining elements
            for (int j = main_end + threadIdx.x; j < M; j += blockDim.x) {
                int idx = row * M + j;
                C[idx] = a_val * B[idx];
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

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Choose approach based on matrix size and alignment
    bool use_vectorized = (M >= 512) && (M % 4 == 0);
    
    if (use_vectorized) {
        const int threads = 256;
        const int blocks = min(65535, (int)((N * M + threads * 4 - 1) / (threads * 4)));
        hybrid_diag_matmul_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            N, M, true);
    } else {
        int threads = (M > 256) ? 256 : (((M + 31) / 32) * 32);
        dim3 grid(N);
        hybrid_diag_matmul_kernel<<<grid, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            N, M, false);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid diagonal matrix multiplication");
}