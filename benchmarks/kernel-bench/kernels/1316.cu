#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use vector types for coalesced memory access
typedef float4 float4_t;

__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const int64_t M_aligned
) {
    // Calculate row this thread block is responsible for
    const int row = blockIdx.x;
    if (row >= N) return;

    // Load diagonal element once and store in register
    const float a_val = A[row];
    
    // Calculate base indices
    const int64_t row_offset = row * M;
    const float4_t* B_vec = reinterpret_cast<const float4_t*>(B + row_offset);
    float4_t* C_vec = reinterpret_cast<float4_t*>(C + row_offset);

    // Process 4 elements at a time using vectorized loads/stores
    // Each thread processes consecutive elements for coalesced access
    for (int j = threadIdx.x; j < M_aligned/4; j += blockDim.x) {
        float4_t b_val = B_vec[j];
        
        // Multiply vector elements by diagonal value
        float4_t c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        
        C_vec[j] = c_val;
    }

    // Handle remaining elements
    const int remaining_start = M_aligned;
    for (int j = remaining_start + threadIdx.x; j < M; j += blockDim.x) {
        C[row_offset + j] = a_val * B[row_offset + j];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    
    // Align M to vector size (float4 = 16 bytes)
    int64_t M_aligned = (M / 4) * 4;

    auto C = torch::empty({N, M}, B.options());

    // Use 128 threads per block for good occupancy while maintaining enough threads
    // for coalescing
    const int threads_per_block = 128;
    const int num_blocks = N;

    diag_matmul_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M,
        M_aligned
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with coalesced memory access");
}