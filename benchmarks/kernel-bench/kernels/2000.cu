#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel maps each thread (1D index) to a unique lower-triangular element in the matrix
// using a closed-form formula. This avoids launching threads that would otherwise process
// upper-triangular (zero) elements, thus improving efficiency.

__global__ void triangular_1d_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N,
                                     int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Map the linear index to (row, col) in the lower triangular part
    // The total number of elements in the lower triangular matrix is N*(N+1)/2.
    // For a given idx, row = floor((sqrt(8*idx + 1)-1)/2) and col = idx - row*(row+1)/2.
    float fidx = (float)idx;
    float r = (sqrtf(8.0f * fidx + 1.0f) - 1.0f) * 0.5f;
    int row = (int)floorf(r);
    int row_start = row * (row + 1) / 2;
    int col = idx - row_start;

    // Perform the triangular matrix multiplication for element (row, col)
    // Since A and B are lower-triangular, only indices k in [col, row] contribute.
    float sum = 0.0f;
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

at::Tensor forward_triangular_1d(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    int total_elements = (N * (N + 1)) / 2; // number of elements in the lower triangular part

    // Initialize output tensor and set upper triangular elements to 0
    auto C = torch::zeros_like(A);

    // Launch a 1D grid covering only the lower triangular elements
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    triangular_1d_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        total_elements
    );
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_triangular_1d, "Triangular matrix multiplication using 1D thread mapping (CUDA)");
}
