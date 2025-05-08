#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    const int col = row + lane_id;

    if (row >= N || col >= N || row > col) return;

    float sum = 0.0f;
    
    // Each thread computes its own summation range from row to col
    for (int k = row; k <= col; ++k) {
        // Let thread with lane_id 0 load A[row,k] and broadcast it
        float a_val = (lane_id == 0) ? A[row * N + k] : 0.0f;
        a_val = __shfl_sync(0xffffffff, a_val, 0);

        float b_val = __ldg(&B[k * N + col]);
        sum += a_val * b_val;
    }

    C[row * N + col] = sum;
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Each block processes 8 rows with 32 threads per row
    dim3 threads(32, 8);
    dim3 blocks(1, (N + threads.y - 1) / threads.y);

    upper_triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Warp-optimized upper triangular matmul");
}