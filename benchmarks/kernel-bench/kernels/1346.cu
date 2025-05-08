#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ldg_aligned_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t vec_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);
    
    for (int idx = tid; idx < vec_elements; idx += stride) {
        // Calculate original matrix indices
        const int base_idx = idx * 4;
        const int row = base_idx / M;
        
        // Use __ldg for read-only data
        const float a_val = __ldg(A + row);
        const float4 b_val = __ldg(&B_vec[idx]);
        
        // Compute output values
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        
        // Store result
        C_vec[idx] = c_val;
    }
    
    // Handle remaining elements if M is not perfectly divisible by 4
    const int remaining_start = (vec_elements * 4);
    if (remaining_start < N * M) {
        for (int i = tid + remaining_start; i < N * M; i += stride) {
            const int row = i / M;
            const float a_val = __ldg(A + row);
            const float b_val = __ldg(B + i);
            C[i] = a_val * b_val;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    // Ensure inputs are contiguous for aligned access
    A = A.contiguous();
    B = B.contiguous();

    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    
    // Create output tensor with same alignment
    auto options = B.options().align_to(16);  // 128-bit alignment
    auto C = torch::empty({N, M}, options);

    // Calculate number of vector elements (float4)
    const int64_t total_elements = N * M;
    const int64_t vec_elements = total_elements / 4;
    
    // Launch kernel with optimal thread configuration
    const int threads = 256;
    const int blocks = min(65535, (int)((vec_elements + threads - 1) / threads));
    
    ldg_aligned_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M,
        vec_elements
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LDG optimized diagonal matrix multiplication");
}