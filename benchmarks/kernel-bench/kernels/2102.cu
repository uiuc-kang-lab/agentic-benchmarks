#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
__device__ inline bool thread_in_bounds(int row, int col, int N) {
    return (row < N && col < N);
}

__device__ inline bool is_lower_triangular(int row, int col) {
    return row >= col;
}

__device__ inline float compute_element(
    const float* __restrict__ A,
    const float* __restrict__ B,
    int row,
    int col,
    int N
) {
    float sum = 0.0f;
    #pragma unroll
    for(int k = col; k <= row; ++k) {
        sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
    }
    return sum;
}
} // anonymous namespace

__global__ void triangular_mm_kernel(
    const float* __restrict__ first_matrix,
    const float* __restrict__ second_matrix,
    float* __restrict__ output_matrix,
    int matrix_size
) {
    const int row = static_cast<int>(blockIdx.y) * blockDim.y + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);

    if(thread_in_bounds(row, col, matrix_size)) {
        if(is_lower_triangular(row, col)) {
            output_matrix[row * matrix_size + col] = compute_element(first_matrix, second_matrix, row, col, matrix_size);
        } else {
            output_matrix[row * matrix_size + col] = 0.0f;
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Invalid matrix dimensions");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    constexpr int threads = 32;
    const dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);
    
    triangular_mm_kernel<<<blocks, dim3(threads, threads)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular triangular matrix multiplication (CUDA)");
}
