#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Optimized kernel for upper triangular matrix multiplication
// Uses strided loops to handle workloads larger than the number of available threads

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride loop to handle larger workloads
    for (int stride_row = row; stride_row < N; stride_row += blockDim.y * gridDim.y) {
        for (int stride_col = col; stride_col < N; stride_col += blockDim.x * gridDim.x) {
            if (stride_row < N && stride_col < N && stride_row <= stride_col) {
                float sum = 0.0f;
                int start_k = stride_row;
                int end_k = stride_col;
                int len = end_k - start_k + 1;

                const float* A_ptr = A + stride_row * N + start_k;
                bool use_vector = false;
                if (len >= 4 && (((uintptr_t)A_ptr) & 0xF) == 0) {
                    use_vector = true;
                }

                int k = start_k;
                if (use_vector) {
                    int vec_iters = len / 4;
                    const float4* A_vec = reinterpret_cast<const float4*>(A_ptr);
                    for (int i = 0; i < vec_iters; i++) {
                        float4 a_val = __ldg(&A_vec[i]);
                        int base_k = start_k + i * 4;
                        sum += a_val.x * __ldg(&B[(base_k) * N + stride_col]);
                        sum += a_val.y * __ldg(&B[(base_k + 1) * N + stride_col]);
                        sum += a_val.z * __ldg(&B[(base_k + 2) * N + stride_col]);
                        sum += a_val.w * __ldg(&B[(base_k + 3) * N + stride_col]);
                    }
                    k = start_k + vec_iters * 4;
                }

                for (; k <= end_k; k++) {
                    float a_val = __ldg(&A[stride_row * N + k]);
                    float b_val = __ldg(&B[k * N + stride_col]);
                    sum += a_val * b_val;
                }
                C[stride_row * N + stride_col] = sum;
            }
        }
    }
}

// Host function that wraps the kernel call
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Strided upper triangular matrix multiplication");
}
