#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel that combines vectorized operations and loop unrolling
__global__ void optimized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;
    if (row < N) {
        float a_val = A[row];
        int offset = row * M;
        int tid = threadIdx.x;
        int stride = blockDim.x;

        if (M % 4 == 0) {
            // Use vectorized operations when M is divisible by 4
            const float4* B_vec = reinterpret_cast<const float4*>(B + offset);
            float4* C_vec = reinterpret_cast<float4*>(C + offset);
            int vec_stride = stride / 4;
            for (int idx = tid; idx < M / 4; idx += vec_stride) {
                float4 b_val = B_vec[idx];
                float4 c_val;
                c_val.x = a_val * b_val.x;
                c_val.y = a_val * b_val.y;
                c_val.z = a_val * b_val.z;
                c_val.w = a_val * b_val.w;
                C_vec[idx] = c_val;
            }
        } else {
            // Use loop unrolling for scalar operations
            const int unroll_factor = 4;
            int step = stride * unroll_factor;
            int main_end = (M / step) * step;
            for (int col = tid; col < main_end; col += step) {
                #pragma unroll
                for (int k = 0; k < unroll_factor; ++k) {
                    int j = col + k * stride;
                    C[offset + j] = a_val * B[offset + j];
                }
            }
            for (int col = main_end + tid; col < M; col += stride) {
                C[offset + col] = a_val * B[offset + col];
            }
        }
    }
}

// Forward function wrapping the optimized CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Launch one block per row, using 256 threads per block
    int threads = 256;
    dim3 grid(N);
    optimized_diag_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiplication with vectorization and loop unrolling");
}