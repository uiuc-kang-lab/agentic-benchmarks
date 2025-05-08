#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

template<int UNROLL_FACTOR = 8>
__global__ void unrolled_upper_triangular_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        int start_k = row;
        int end_k = col;
        
        // Vectorized processing with aggressive unrolling
        const float* A_row = A + row * N;
        const float* B_col = B + col;
        
        // Process vectors of 4 elements
        int k = start_k;
        int vec_end = end_k - ((end_k - start_k + 1) % (4 * UNROLL_FACTOR));
        
        #pragma unroll
        for (; k <= vec_end; k += 4 * UNROLL_FACTOR) {
            float4 a_vals[UNROLL_FACTOR];
            float4 b_vals[UNROLL_FACTOR];
            
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                int idx = k + u * 4;
                const float4* a_ptr = reinterpret_cast<const float4*>(A_row + idx - (idx % 4));
                const float4* b_ptr = reinterpret_cast<const float4*>(B_col + idx * N);
                
                a_vals[u] = __ldg(a_ptr);
                b_vals[u] = __ldg(b_ptr);
                
                sum += a_vals[u].x * b_vals[u].x;
                sum += a_vals[u].y * b_vals[u].y;
                sum += a_vals[u].z * b_vals[u].z;
                sum += a_vals[u].w * b_vals[u].w;
            }
        }
        
        // Handle remaining elements
        #pragma unroll 4
        for (; k <= end_k; k++) {
            sum += __ldg(&A_row[k]) * __ldg(&B[k * N + col]);
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    unrolled_upper_triangular_kernel<8><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Unrolled upper triangular matrix multiplication");
}