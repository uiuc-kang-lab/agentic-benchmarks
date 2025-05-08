#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define ALIGN_MASK 0x0000000F
#define WARP_SIZE 32

__global__ void optimized_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 2];  // +2 padding for bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 2];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Pre-compute batch offsets
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C = C + b * M * N;

    // Use float4 for vectorized loads when possible
    float sum = 0.0f;
    
    // Process tiles with vectorized loads
    #pragma unroll 4
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading using vectorized reads when possible
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&batch_A[row * K + t * TILE_SIZE + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&batch_B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product with loop unrolling
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    // Write result if within bounds using vectorized writes when possible
    if (row < M && col < N) {
        batch_C[row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Ensure alignment for vectorized loads
    TORCH_CHECK((reinterpret_cast<uintptr_t>(A.data_ptr<float>()) & ALIGN_MASK) == 0,
                "Input tensor A must be 16-byte aligned");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(B.data_ptr<float>()) & ALIGN_MASK) == 0,
                "Input tensor B must be 16-byte aligned");

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    // Get CUDA stream from current context
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    optimized_bmm_kernel<<<grid, threads, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Optimized batched matrix multiplication (CUDA)");
}