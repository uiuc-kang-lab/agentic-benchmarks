#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int WARP_SIZE = 32>
__global__ void warp_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_count = blockDim.x / WARP_SIZE;
    const unsigned int global_warp_id = blockIdx.x * warp_count + warp_id;
    
    // Calculate which row this warp is processing
    const int row = global_warp_id;
    
    if (row < N) {
        // First lane in warp loads the diagonal element
        float a_val = (lane_id == 0) ? A[row] : 0.0f;
        
        // Broadcast a_val to all lanes in the warp using shuffle
        a_val = __shfl_sync(0xffffffff, a_val, 0);
        
        // Process elements in chunks of 4 (float4) for better memory coalescing
        const int vec_elements = M / 4;
        const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
        float4* C_vec = reinterpret_cast<float4*>(C + row * M);
        
        // Each lane processes multiple float4 elements
        for (int idx = lane_id; idx < vec_elements; idx += WARP_SIZE) {
            float4 b_val = B_vec[idx];
            float4 c_val;
            c_val.x = a_val * b_val.x;
            c_val.y = a_val * b_val.y;
            c_val.z = a_val * b_val.z;
            c_val.w = a_val * b_val.w;
            C_vec[idx] = c_val;
        }
        
        // Handle remaining elements
        const int vec_processed = vec_elements * 4;
        for (int idx = vec_processed + lane_id; idx < M; idx += WARP_SIZE) {
            C[row * M + idx] = a_val * B[row * M + idx];
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Configure kernel launch parameters
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    const int threads_per_block = WARP_SIZE * WARPS_PER_BLOCK;
    const int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    warp_diag_matmul_kernel<WARP_SIZE><<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized diagonal matrix multiplication");
}