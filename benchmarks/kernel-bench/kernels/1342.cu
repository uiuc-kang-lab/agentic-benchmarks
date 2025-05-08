#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory to cache diagonal values
__global__ void shared_memory_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M) {
    extern __shared__ float shared_A[];  // Shared memory for diagonal values

    int row = blockIdx.x;
    if (row >= N) return;

    // Load diagonal value into shared memory
    if (threadIdx.x == 0) {
        shared_A[0] = A[row];
    }
    __syncthreads();

    float a_val = shared_A[0];
    int offset = row * M;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Vectorized processing using float4 if M is divisible by 4
    if (M % 4 == 0) {
        const float4* B_vec = reinterpret_cast<const float4*>(B + offset);
        float4* C_vec = reinterpret_cast<float4*>(C + offset);
        int vec_M = M >> 2;  // Divide by 4

        for (int vid = tid; vid < vec_M; vid += stride) {
            float4 b_val = B_vec[vid];
            float4 c_val;
            c_val.x = a_val * b_val.x;
            c_val.y = a_val * b_val.y;
            c_val.z = a_val * b_val.z;
            c_val.w = a_val * b_val.w;
            C_vec[vid] = c_val;
        }

        // Handle remaining elements
        int remaining_start = vec_M * 4;
        for (int col = remaining_start + tid; col < M; col += stride) {
            C[offset + col] = a_val * B[offset + col];
        }
    } else {
        // Scalar processing for non-vectorized case
        for (int col = tid; col < M; col += stride) {
            C[offset + col] = a_val * B[offset + col];
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

    int threads = 256;
    dim3 grid(N);
    size_t shared_mem_size = sizeof(float);  // Shared memory size for one float

    shared_memory_diag_matmul_kernel<<<grid, threads, shared_mem_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory optimized diagonal matrix multiplication");
}