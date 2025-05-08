#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<bool UseVector>
__global__ void shared_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t elements_per_thread
) {
    __shared__ float shared_diag;  // Cache diagonal value in shared memory
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int global_idx = bid * num_threads + tid;
    
    if (UseVector) {
        // Vectorized version
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);
        const int vec_M = M >> 2;  // M/4
        
        for (int i = global_idx; i < N * vec_M; i += gridDim.x * num_threads) {
            const int row = i / vec_M;
            
            // First thread in block loads diagonal value
            if (tid == 0) {
                shared_diag = A[row];
            }
            __syncthreads();
            
            float4 b_val = B_vec[i];
            float4 c_val;
            c_val.x = shared_diag * b_val.x;
            c_val.y = shared_diag * b_val.y;
            c_val.z = shared_diag * b_val.z;
            c_val.w = shared_diag * b_val.w;
            C_vec[i] = c_val;
            
            __syncthreads();
        }
    } else {
        // Scalar version
        for (int base = global_idx; base < N * M; base += gridDim.x * num_threads) {
            const int row = base / M;
            
            // First thread in block loads diagonal value
            if (tid == 0) {
                shared_diag = A[row];
            }
            __syncthreads();
            
            // Process elements_per_thread elements per thread
            #pragma unroll 4
            for (int offset = 0; offset < elements_per_thread && (base + offset) < N * M; offset++) {
                const int idx = base + offset;
                if (idx < N * M) {
                    C[idx] = shared_diag * B[idx];
                }
            }
            
            __syncthreads();
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

    const int threads = 256;
    const int elements_per_thread = 4;
    
    if (M % 4 == 0) {
        // Use vectorized version for aligned data
        const int blocks = min(65535, (int)((N * M + threads * 4 - 1) / (threads * 4)));
        shared_diag_matmul_kernel<true><<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N, M,
            elements_per_thread
        );
    } else {
        // Use scalar version for unaligned data
        const int blocks = min(65535, (int)((N * M + threads * elements_per_thread - 1) / (threads * elements_per_thread)));
        shared_diag_matmul_kernel<false><<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N, M,
            elements_per_thread
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory diagonal matrix multiplication");
}