#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel computes C = A * B for lower triangular matrices A and B,
// where C[i,j] = \sum_{k=j}^{i} A[i,k]*B[k,j] for i>=j, and 0 otherwise.
// The kernel uses warp-level reduction: each warp (32 threads) cooperates to compute one lower triangular output element.
// Then, a grid-stride loop fills in the upper triangular elements with zero.

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N,
                                     int lower_count) {
    // Total number of elements in the matrix
    int total_matrix = N * N;
    
    // Each lower triangular element (one per warp) uses 32 threads.
    // Total threads needed for lower triangular computation
    int total_lower_threads = lower_count * 32;

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    // --- Lower triangular part computation ---
    // Each warp (32 consecutive threads) computes one output element in the lower triangle.
    // We map a warpId to a (i, j) using the formula: warpId = i*(i+1)/2 + j, where 0 <= j <= i < N.
    for (int global_id = tid; global_id < total_lower_threads; global_id += num_threads) {
        int warpId = global_id / 32;
        int lane = global_id % 32;

        // Convert warpId to (i,j) using the inverse of i*(i+1)/2.
        // i = floor((sqrt(8*warpId+1)-1)/2) and j = warpId - i*(i+1)/2
        float f = __fsqrt_rn(8.0f * warpId + 1.0f);
        int i = (int)((f - 1.0f) * 0.5f);  // floor((sqrt(8*warpId+1)-1)/2)
        int temp = i * (i + 1) / 2;
        int j = warpId - temp;

        // Ensure indices are within the matrix. (They should be if warpId < lower_count.)
        if(i < N && j < N) {
            // The summation runs from k = j to k = i.
            int L = i - j + 1;
            float partial = 0.0f;
            
            // Each lane of the warp handles a subset of the k range in a strided manner.
            for (int offset = lane; offset < L; offset += 32) {
                int k = j + offset;
                // k <= i by construction and we check boundaries
                if (k < N) {
                    partial += A[i * N + k] * B[k * N + j];
                }
            }
            
            // Perform warp-level reduction using __shfl_down_sync
            unsigned mask = 0xffffffff;
            for (int offset = 16; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(mask, partial, offset);
            }
            
            // The first lane writes the result for the lower triangular element C[i,j]
            if (lane == 0) {
                C[i * N + j] = partial;
            }
        }
    }
    
    // --- Upper triangular part: fill with zeros ---
    // Use a grid-stride loop over the entire matrix and set C[i,j] = 0 for i < j.
    for (int idx = tid; idx < total_matrix; idx += num_threads) {
        int i = idx / N;
        int j = idx % N;
        if (i < j) {
            C[i * N + j] = 0.0f;
        }
    }
}

// C++ interface exposed to PyTorch.
// This function computes the matrix multiplication for lower triangular matrices A and B, returning the result in C.

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Number of lower triangular elements
    int lower_count = (N * (N + 1)) / 2;
    
    // Determine the total number of threads needed.
    // For lower-triangular part, each of the lower_count output elements is computed by 32 threads.
    int total_lower_threads = lower_count * 32;
    int total_matrix = N * N;
    int total_threads = total_lower_threads > total_matrix ? total_lower_threads : total_matrix;

    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    triangular_mm_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        lower_count
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level reduced triangular matrix multiplication (CUDA)");
}
