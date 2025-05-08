#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Constant memory for lookup table - limited to 64KB
__constant__ int const_lookup[1024];  // Adjust size based on expected matrix dimensions

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        int start_k = row;
        int end_k = col;
        int len = end_k - start_k + 1;

        const float* A_ptr = A + row * N + start_k;
        bool use_vector = (len >= 4 && (((uintptr_t)A_ptr) & 0xF) == 0);

        int k = start_k;
        if (use_vector) {
            int vec_iters = len / 4;
            const float4* A_vec = reinterpret_cast<const float4*>(A_ptr);
            
            #pragma unroll 4
            for (int i = 0; i < vec_iters; i++) {
                float4 a_val = __ldg(&A_vec[i]);
                int base_k = start_k + i * 4;
                
                // Use constant memory for index calculations
                int idx = base_k & 1023;  // Modulo 1024 for lookup table
                int offset = const_lookup[idx];
                
                sum += a_val.x * __ldg(&B[(base_k + offset) * N + col]);
                sum += a_val.y * __ldg(&B[(base_k + 1) * N + col]);
                sum += a_val.z * __ldg(&B[(base_k + 2) * N + col]);
                sum += a_val.w * __ldg(&B[(base_k + 3) * N + col]);
            }
            k = start_k + vec_iters * 4;
        }

        #pragma unroll 4
        for (; k <= end_k; k++) {
            float a_val = __ldg(&A[row * N + k]);
            float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Initialize constant memory lookup table
    int host_lookup[1024];
    for(int i = 0; i < 1024; i++) {
        host_lookup[i] = i % 4;  // Example pattern for lookup
    }
    cudaMemcpyToSymbol(const_lookup, host_lookup, sizeof(int) * 1024);

    // Using a 16x16 block size (256 threads per block), a multiple of warp size
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    upper_triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}