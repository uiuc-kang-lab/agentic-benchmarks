#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Macro checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define tile size and split_k factor
#define TILE_DIM 32
#define SPLIT_K 8

// This kernel partitions the reduction over the K dimension (split-k) and uses atomicAdd
// only once per thread to accumulate its partial sum into global memory. This minimizes
// atomic usage to the minimum necessary to handle concurrent updates when multiple blocks
// contribute to the same output element.

__global__ void splitk_matrix_mult_atomic_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   const int M, const int N, const int K) {
    // Determine the portion of K this block will process
    int split_k = gridDim.z; // equals SPLIT_K
    int tile_k = (K + split_k - 1) / split_k; // size of each K-partition
    int k_start = blockIdx.z * tile_k;
    int k_end = (k_start + tile_k < K) ? (k_start + tile_k) : K;

    // Identify the output element (row, col) computed by this thread
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Allocate shared memory once for the block
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over the current partition in steps of TILE_DIM
    for (int t = k_start; t < k_end; t += TILE_DIM) {
        // Compute the effective tile width (handles edge case at partition end)
        int effective_tile = (t + TILE_DIM <= k_end) ? TILE_DIM : (k_end - t);

        // Load tile of A into shared memory
        if (row < M && (t + threadIdx.x) < k_end) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        if ((t + threadIdx.y) < k_end && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for the loaded tile
        for (int i = 0; i < effective_tile; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Use atomicAdd only once per thread to accumulate the partial sum
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}


// Host function to launch the kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM,
              SPLIT_K);

    splitk_matrix_mult_atomic_kernel<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
}

// Forward interface called from Python
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Initialize C to zero because atomicAdd will accumulate partial sums
    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with split-K and atomic additions (CUDA)");
}
