#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for warp-level vectorized matrix multiplication
// Each warp computes one element of output C = A * B
// A is MxK and stored in row-major order, B is KxN stored in row-major order.
// The kernel uses vectorized loads (float4) for A and manual unrolling for B.
// Partial dot-product results are reduced within the warp using __shfl_down_sync().

__global__ void warp_vectorized_matmul_kernel(const float* __restrict__ A, 
                                                const float* __restrict__ B, 
                                                float* __restrict__ C, 
                                                int M, int N, int K) {
    // Each warp (32 threads) computes one output element in C
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;  // one warp per C element
    int lane = tid % 32;

    int total_elems = M * N;
    if (warp_id >= total_elems) return;

    // Determine target row and column in output matrix C
    int row = warp_id / N;
    int col = warp_id % N;

    float sum = 0.0f;

    // Process the K-dimension in chunks using vectorized loads for A
    int vecK = K / 4;  // number of groups of 4

    // Pointer to the beginning of A's row
    const float* A_row = A + row * K;
    // Reinterpret the row as float4 for vectorized loads
    const float4* A_row_vec = reinterpret_cast<const float4*>(A_row);

    // Each lane processes a strided portion of the vectorized K dimension
    for (int i = lane; i < vecK; i += 32) {
        int k_index = i * 4;  // starting index in the K dimension for this group
        float4 a_val = A_row_vec[i];
        
        // For B, since elements are not contiguous (stride is N), load each element individually
        float b0 = B[(k_index + 0) * N + col];
        float b1 = B[(k_index + 1) * N + col];
        float b2 = B[(k_index + 2) * N + col];
        float b3 = B[(k_index + 3) * N + col];
        
        float partial = a_val.x * b0 + a_val.y * b1 + a_val.z * b2 + a_val.w * b3;
        sum += partial;
    }

    // Process remaining elements in K (if K is not a multiple of 4)
    for (int k = vecK * 4 + lane; k < K; k += 32) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // Warp-level reduction using shuffle intrinsics
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the result for this output element
    if (lane == 0) {
        C[row * N + col] = sum;
    }
}


void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Each output element is computed by one warp (32 threads)
    int total_elems = M * N;
    int total_threads = total_elems * 32;

    // Use a block size of 256 threads
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    warp_vectorized_matmul_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


// The forward function allocates the output tensor and calls our custom matrix multiplication kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level vectorized matrix multiplication with shuffle reduction (CUDA)");
}
