#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// by flattening the work to cover only valid (row, col) pairs in the upper triangle.
// The total number of valid elements is T = N*(N+1)/2. Each thread computes its
// global 1D index (tid) and iterates over the flattened indices with a given stride.
// The mapping from a flattened index idx to (row, col) is computed using the quadratic
// formula: 
//   r = floor(((2*N + 1) - sqrt((2*N+1)^2 - 8*idx)) / 2)
//   Fr = r * N - (r*(r-1))/2  (starting index of row r in flattened order)
//   c = idx - Fr + r
// Each thread then computes the dot product for element (r, c) over k from r to c.
// This approach evenly distributes the workload among threads and avoids idle threads
// from conditional tests in a 2D grid mapping.

__global__ void flattened_upper_triangular_kernel_1D(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int N,
                                                      int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        // Map flattened index to (row, col) in the upper triangular matrix
        float term = sqrtf((2.0f * N + 1.0f) * (2.0f * N + 1.0f) - 8.0f * (float)idx);
        int r = (int)floor(((2.0f * N + 1.0f) - term) / 2.0f);
        int Fr = r * N - (r * (r - 1)) / 2;  // starting index for row r
        int c = idx - Fr + r;               // column index for the flattened idx
        
        float sum = 0.0f;
        for (int k = r; k <= c; ++k) {
            sum += A[r * N + k] * B[k * N + c];
        }
        C[r * N + c] = sum;
    }
}

torch::Tensor flattened_upper_triangular_matmul_forward(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    // Total number of elements in the upper triangular (including the diagonal)
    int total_elements = N * (N + 1) / 2;
    auto C = torch::zeros_like(A);

    // Choose a 1D configuration that evenly distributes work.
    const int threadsPerBlock = 256;
    int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    flattened_upper_triangular_kernel_1D<<<numBlocks, threadsPerBlock>>>(
         A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, total_elements);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flattened_upper_triangular_matmul_forward, "Flattened Upper Triangular Matrix Multiplication with Even Work Distribution");
}
