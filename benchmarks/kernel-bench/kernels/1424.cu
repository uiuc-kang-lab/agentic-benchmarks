#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define VECTOR_SIZE 4
#define SHARED_DIAG_SIZE 128

__device__ __forceinline__ float4 load_vector_coalesced(const float* addr) {
    return *reinterpret_cast<const float4*>(addr + threadIdx.x * VECTOR_SIZE);
}

__device__ __forceinline__ void store_vector_coalesced(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr + threadIdx.x * VECTOR_SIZE) = val;
}

__global__ void diag_matmul_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    __shared__ float shared_diag[SHARED_DIAG_SIZE];
    
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int local_warp = threadIdx.x / WARP_SIZE;

    // Cache diagonal values in shared memory
    if (threadIdx.x < SHARED_DIAG_SIZE && blockIdx.x * SHARED_DIAG_SIZE + threadIdx.x < N) {
        shared_diag[threadIdx.x] = A[blockIdx.x * SHARED_DIAG_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (warp_id < N) {
        const float a_val = __shfl_sync(0xffffffff, shared_diag[local_warp], 0);
        const int row = warp_id;
        const int row_offset = row * M;

        if (M % VECTOR_SIZE == 0) {
            const int vec_M = M / VECTOR_SIZE;
            for (int base = 0; base < vec_M; base += blockDim.x) {
                const int vec = base + threadIdx.x;
                if (vec < vec_M) {
                    float4 b_vec = load_vector_coalesced(&B[row_offset + vec * VECTOR_SIZE]);
                    b_vec.x *= a_val;
                    b_vec.y *= a_val;
                    b_vec.z *= a_val;
                    b_vec.w *= a_val;
                    store_vector_coalesced(&C[row_offset + vec * VECTOR_SIZE], b_vec);
                }
            }
        } else {
            for (int base = 0; base < M; base += blockDim.x) {
                const int col = base + threadIdx.x;
                if (col < M) {
                    C[row_offset + col] = a_val * B[row_offset + col];
                }
            }
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

    const int threads = 256;
    const int blocks = (N + (SHARED_DIAG_SIZE - 1)) / SHARED_DIAG_SIZE;
    
    diag_matmul_kernel_optimized<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matmul with shared mem and coalesced access");
}