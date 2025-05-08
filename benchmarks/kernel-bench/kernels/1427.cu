#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define VECTOR_SIZE 4

// Device function to broadcast the diagonal element using warp-level primitives
__device__ __inline__ float broadcast_diag(const float* __restrict__ A, int warp_id) {
    return __shfl_sync(0xffffffff, A[warp_id], 0);
}

// Device function to process a row using vectorized memory operations; assumes M is divisible by VECTOR_SIZE
__device__ __inline__ void process_vectorized(const float diag_val, const float* __restrict__ B_row, float* __restrict__ C_row, int lane, int M) {
    int vec_M = M / VECTOR_SIZE;
    for (int vec = lane; vec < vec_M; vec += WARP_SIZE) {
        int idx = vec * VECTOR_SIZE;  // starting index for this vector block
        // Load a vector of 4 floats from B
        float4 b_vals = *reinterpret_cast<const float4*>(&B_row[idx]);
        float4 c_vals;
        c_vals.x = diag_val * b_vals.x;
        c_vals.y = diag_val * b_vals.y;
        c_vals.z = diag_val * b_vals.z;
        c_vals.w = diag_val * b_vals.w;
        // Store the result into C
        *reinterpret_cast<float4*>(&C_row[idx]) = c_vals;
    }
}

// Device function to process a row scalar-wise
__device__ __inline__ void process_scalar(const float diag_val, const float* __restrict__ B_row, float* __restrict__ C_row, int lane, int M) {
    for (int col = lane; col < M; col += WARP_SIZE) {
        C_row[col] = __fmul_rn(diag_val, B_row[col]);
    }
}

// Device function to process a row, choosing vectorized or scalar based on M
__device__ __inline__ void process_row(const float diag_val, const float* __restrict__ B_row, float* __restrict__ C_row, int M, int lane) {
    if (M % VECTOR_SIZE == 0) {
        process_vectorized(diag_val, B_row, C_row, lane, M);
    } else {
        process_scalar(diag_val, B_row, C_row, lane, M);
    }
}

// CUDA kernel: each warp processes one row of the matrix multiplication
__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    
    if (warpId < N) {
        // Broadcast diagonal value for the current row
        float diag_val = broadcast_diag(A, warpId);
        
        // Get pointers to the current row in B and C
        const float* B_row = &B[warpId * M];
        float* C_row = &C[warpId * M];
        
        // Process the row in a modular fashion
        process_row(diag_val, B_row, C_row, M, lane);
    }
}

// Forward function wrapping the CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Configure kernel launch: assign one warp per row
    int threadsPerBlock = 128; // Must be a multiple of WARP_SIZE
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with modular device functions");
}
