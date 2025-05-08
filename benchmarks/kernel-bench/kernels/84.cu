#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 32  // Tile size in shared memory

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// Constant memory for the matrix dimension
__constant__ int const_N;

// This kernel uses register tiling where each thread computes a 2x2 submatrix of C.
// The matrix dimension is stored in constant memory to reduce global memory traffic.
__global__ void const_regtile_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each block is launched with 16x16 threads; each thread computes a 2x2 submatrix.
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int row = blockIdx.y * BLOCK_SIZE + ty * 2;  // row index for 2 rows
    int col = blockIdx.x * BLOCK_SIZE + tx * 2;    // col index for 2 columns

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Loop over tiles along k-dimension
    for (int m = 0; m < const_N; m += BLOCK_SIZE) {
        int threadsPerBlock = blockDim.x * blockDim.y; // 256 threads per block
        int index = threadIdx.y * blockDim.x + threadIdx.x;

        // Load tile of matrix A into shared memory
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / threadsPerBlock; i++) {
            int loadIndex = index + i * threadsPerBlock;
            int ar = loadIndex / BLOCK_SIZE;
            int ac = loadIndex % BLOCK_SIZE;
            int global_r = blockIdx.y * BLOCK_SIZE + ar;
            int global_c = m + ac;
            if (global_r < const_N && global_c < const_N)
                As[ar][ac] = A[global_r * const_N + global_c];
            else
                As[ar][ac] = 0.0f;
        }

        // Load tile of matrix B into shared memory
        for (int i = 0; i < (BLOCK_SIZE * BLOCK_SIZE) / threadsPerBlock; i++) {
            int loadIndex = index + i * threadsPerBlock;
            int br = loadIndex / BLOCK_SIZE;
            int bc = loadIndex % BLOCK_SIZE;
            int global_r = m + br;
            int global_c = blockIdx.x * BLOCK_SIZE + bc;
            if (global_r < const_N && global_c < const_N)
                Bs[br][bc] = B[global_r * const_N + global_c];
            else
                Bs[br][bc] = 0.0f;
        }
        __syncthreads();

        // Compute partial results for the 2x2 submatrix that this thread is responsible for
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
        __syncthreads();
    }

    // Write the computed 2x2 submatrix back to global memory with boundary checking
    if (row < const_N && col < const_N)
        C[row * const_N + col] = c00;
    if (row < const_N && (col + 1) < const_N)
        C[row * const_N + col + 1] = c01;
    if ((row + 1) < const_N && col < const_N)
        C[(row + 1) * const_N + col] = c10;
    if ((row + 1) < const_N && (col + 1) < const_N)
        C[(row + 1) * const_N + col + 1] = c11;
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

    // Copy the matrix size into constant memory
    cudaMemcpyToSymbol(const_N, &N, sizeof(int), cudaMemcpyHostToDevice);

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Launch configuration: 16x16 threads per block (each thread computes a 2x2 submatrix)
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const_regtile_matmul_kernel<<<blocks, threads>>>(A_data, B_data, C_data);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with constant memory for read-only data");
}
