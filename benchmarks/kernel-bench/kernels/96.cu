#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 32  // Tile size for shared memory

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// This kernel uses register tiling and a 2x2 submatrix per thread.
// Each block computes a 32x32 tile of C using a thread block of 16x16 threads,
// where each thread computes 2 rows and 2 columns of the output using registers.

__global__ void optimized_thread_block_mapping_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    // Shared memory tiles for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each block is launched with 16x16 threads
    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    // Compute the starting indices of the 2x2 submatrix computed by this thread
    int row = blockIdx.y * BLOCK_SIZE + ty * 2;  // each thread covers 2 rows
    int col = blockIdx.x * BLOCK_SIZE + tx * 2;    // each thread covers 2 columns

    // Registers to accumulate the 2x2 block
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Loop over tiles along the K dimension
    for (int m = 0; m < N; m += BLOCK_SIZE) {
        // Load tile of A into shared memory
        // There are BLOCK_SIZE*BLOCK_SIZE elements, and 16x16 = 256 threads per block.
        // Each thread loads 4 elements to cover the entire tile.
        int threadsPerBlock = blockDim.x * blockDim.y; // 256
        int index = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / threadsPerBlock; i++) {
            int loadIndex = index + i * threadsPerBlock;
            int ar = loadIndex / BLOCK_SIZE;
            int ac = loadIndex % BLOCK_SIZE;
            int global_r = blockIdx.y * BLOCK_SIZE + ar;
            int global_c = m + ac;
            if (global_r < N && global_c < N) {
                As[ar][ac] = A[global_r * N + global_c];
            } else {
                As[ar][ac] = 0.0f;
            }
        }

        // Load tile of B into shared memory
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / threadsPerBlock; i++) {
            int loadIndex = index + i * threadsPerBlock;
            int br = loadIndex / BLOCK_SIZE;
            int bc = loadIndex % BLOCK_SIZE;
            int global_r = m + br;
            int global_c = blockIdx.x * BLOCK_SIZE + bc;
            if (global_r < N && global_c < N) {
                Bs[br][bc] = B[global_r * N + global_c];
            } else {
                Bs[br][bc] = 0.0f;
            }
        }
        if (k < BLOCK_SIZE) __syncthreads();

        // Compute the 2x2 submatrix for this thread
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        if (k < BLOCK_SIZE) __syncthreads();
    }

    // Write the computed 2x2 submatrix back to global memory with boundary checks
    if (row < N && col < N)
        C[row * N + col] = c00;
    if (row < N && (col + 1) < N)
        C[row * N + col + 1] = c01;
    if ((row + 1) < N && col < N)
        C[(row + 1) * N + col] = c10;
    if ((row + 1) < N && (col + 1) < N)
        C[(row + 1) * N + col + 1] = c11;
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Each block covers a 32x32 tile, with 16x16 threads (each thread computes a 2x2 submatrix)
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    optimized_thread_block_mapping_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized thread and block mapping matrix multiplication kernel (CUDA)");
}
