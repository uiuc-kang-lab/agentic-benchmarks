#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __inline__ float4 load_vector(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__device__ __inline__ void store_vector(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

__device__ __inline__ float broadcast_diag_value(int warp_id, const float* A) {
    return __shfl_sync(0xffffffff, A[warp_id], 0);
}

__device__ __inline__ void process_vectorized_columns(
    float a_val,
    const float* B_row,
    float* C_row,
    int lane,
    int M
) {
    const int vec_size = 4;
    const int vec_M = M / vec_size;
    for (int vec = lane; vec < vec_M; vec += WARP_SIZE) {
        float4 b_vec = load_vector(&B_row[vec * vec_size]);
        float4 c_vec;
        c_vec.x = a_val * b_vec.x;
        c_vec.y = a_val * b_vec.y;
        c_vec.z = a_val * b_vec.z;
        c_vec.w = a_val * b_vec.w;
        store_vector(&C_row[vec * vec_size], c_vec);
    }
}

__device__ __inline__ void process_scalar_columns(
    float a_val,
    const float* B_row,
    float* C_row,
    int lane,
    int M
) {
    for (int col = lane; col < M; col += WARP_SIZE) {
        C_row[col] = a_val * B_row[col];
    }
}

__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    if (warp_id < N) {
        const float a_val = broadcast_diag_value(warp_id, A);
        const float* B_row = &B[warp_id * M];
        float* C_row = &C[warp_id * M];

        if (M % 4 == 0) {
            process_vectorized_columns(a_val, B_row, C_row, lane, M);
        } else {
            process_scalar_columns(a_val, B_row, C_row, lane, M);
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

    int threads = 128;
    int blocks = (N * WARP_SIZE + threads - 1) / threads;
    diag_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matmul with modular vector ops");
}
