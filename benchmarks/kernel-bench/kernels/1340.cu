#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel combining vectorized and row-based approaches
__global__ void adaptive_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t strategy  // 0: vectorized, 1: row-based small, 2: flat scalar
) {
    if (strategy == 0) {
        // Vectorized approach for large aligned matrices
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        const int vec_total = (N * M) / 4;
        
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
    } 
    else if (strategy == 1) {
        // Row-based approach for smaller matrices
        int row = blockIdx.x;
        if (row < N) {
            float a_val = A[row];
            
            // Use shared memory for frequently accessed a_val
            __shared__ float shared_a;
            if (threadIdx.x == 0) shared_a = a_val;
            __syncthreads();
            
            const int main_end = (M / blockDim.x) * blockDim.x;
            
            // Coalesced main loop with vectorized loads where possible
            for (int j = threadIdx.x; j < main_end; j += blockDim.x) {
                int idx = row * M + j;
                C[idx] = shared_a * B[idx];
            }
            
            // Handle remaining elements
            for (int j = main_end + threadIdx.x; j < M; j += blockDim.x) {
                int idx = row * M + j;
                C[idx] = shared_a * B[idx];
            }
        }
    }
    else {
        // Flat scalar approach for medium-sized or unaligned matrices
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        const int total = N * M;
        
        for (; idx < total; idx += stride) {
            int row = idx / M;
            C[idx] = A[row] * B[idx];
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

    // Adaptive strategy selection based on matrix characteristics
    int strategy;
    dim3 blocks, threads;
    
    if (M >= 512 && M % 4 == 0) {
        // Large aligned matrices: use vectorized approach
        strategy = 0;
        threads = dim3(256);
        blocks = dim3(min(65535, (int)((N * M + threads.x * 4 - 1) / (threads.x * 4))));
    }
    else if (N <= 256 && M <= 1024) {
        // Small matrices: use row-based approach
        strategy = 1;
        threads = dim3(min(256, (int)(((M + 31) / 32) * 32)));
        blocks = dim3(N);
    }
    else {
        // Medium or unaligned matrices: use flat scalar approach
        strategy = 2;
        threads = dim3(256);
        blocks = dim3((N * M + threads.x - 1) / threads.x);
    }

    adaptive_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M, strategy);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive diagonal matrix multiplication");
}