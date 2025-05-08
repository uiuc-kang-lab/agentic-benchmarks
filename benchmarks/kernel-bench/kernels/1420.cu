#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define VECTOR_SIZE 4

template<typename T>
__device__ __forceinline__ T load_broadcast_diagonal(const T* __restrict__ A, int warpId) {
    T a_val = __shfl_sync(0xffffffff, A[warpId], 0);
    return a_val;
}

template<typename T>
__device__ __forceinline__ void process_vectorized(
    const T* __restrict__ B,
    T* __restrict__ C,
    const T diag_val,
    const int row,
    const int vec_M,
    const int lane
) {
    using Vec4 = typename std::conditional<std::is_same<T, float>::value, float4, double4>::type;
    
    #pragma unroll
    for (int vec = lane; vec < vec_M; vec += WARP_SIZE) {
        int idx = row * vec_M + vec;
        Vec4 b_val = reinterpret_cast<const Vec4*>(B)[idx];
        Vec4 c_val;
        c_val.x = diag_val * b_val.x;
        c_val.y = diag_val * b_val.y;
        c_val.z = diag_val * b_val.z;
        c_val.w = diag_val * b_val.w;
        reinterpret_cast<Vec4*>(C)[idx] = c_val;
    }
}

template<typename T>
__device__ __forceinline__ void process_scalar(
    const T* __restrict__ B,
    T* __restrict__ C,
    const T diag_val,
    const int row_offset,
    const int M,
    const int lane
) {
    for (int col = lane; col < M; col += WARP_SIZE) {
        int idx = row_offset + col;
        C[idx] = diag_val * B[idx];
    }
}

__global__ void diag_matmul_kernel_modular(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    if (warpId < N) {
        const float diag_val = load_broadcast_diagonal(A, warpId);
        const int row = warpId;
        const int row_offset = row * M;

        if (M % VECTOR_SIZE == 0) {
            process_vectorized(B, C, diag_val, row, M/VECTOR_SIZE, lane);
        } else {
            process_scalar(B, C, diag_val, row_offset, M, lane);
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_modular<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using modular device functions");
}