#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<bool USE_VECTORIZED>
__global__ void diag_matmul_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M) {
    
    if (USE_VECTORIZED) {
        const int64_t vec_size = 4;
        const int64_t vec_M = M / vec_size;
        const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        const float4* B_vec = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
        float4* C_vec = reinterpret_cast<float4*>(__builtin_assume_aligned(C, 16));
        
        for(int64_t idx = global_idx; idx < N * vec_M; idx += gridDim.x * blockDim.x) {
            const int row = idx / vec_M;
            const int vec_col = idx % vec_M;
            
            const float a_val = __ldg(A + row);
            const float4 b_val = __ldg(B_vec + row * vec_M + vec_col);
            
            float4 c_val;
            c_val.x = a_val * b_val.x;
            c_val.y = a_val * b_val.y;
            c_val.z = a_val * b_val.z;
            c_val.w = a_val * b_val.w;
            
            C_vec[row * vec_M + vec_col] = c_val;
        }
    } else {
        const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        for(int64_t idx = global_idx; idx < N * M; idx += gridDim.x * blockDim.x) {
            const int row = idx / M;
            const int col = idx % M;
            C[idx] = __ldg(A + row) * __ldg(B + idx);
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

    constexpr int threads = 256;
    const int blocks = (N * M + threads - 1) / threads;

    if (M % 4 == 0) {
        diag_matmul_optimized_kernel<true><<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M
        );
    } else {
        diag_matmul_optimized_kernel<false><<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            M
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matmul with LDG and aligned accesses");
}