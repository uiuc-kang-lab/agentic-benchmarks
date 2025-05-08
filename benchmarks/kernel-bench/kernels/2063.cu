#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for computing C = A * B, where A and B are lower triangular matrices.
// Each thread computes one element of C, and we use a tunable block (tile) size for experimentation.
__global__ void tunable_block_triangular_mm_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N) {
    // Compute the row and column indices using block dimensions.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Only indices from col to row contribute for lower triangular matrices.
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch. The third parameter, tile_size, allows experimentation with block sizes.
// Typical tile_size values (for each dimension) might be chosen so that total threads per block are
// 32, 64, 128, 256, 512 etc. For example, tile_size=8 gives 64 threads; tile_size=16 gives 256 threads.

at::Tensor forward(at::Tensor A, at::Tensor B, int tile_size = 16) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Define block and grid dimensions based on the tunable tile size.
    // Using a 2D configuration with each block having (tile_size x tile_size) threads.
    dim3 block(tile_size, tile_size);
    dim3 grid((N + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);

    tunable_block_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tunable block size triangular matrix multiplication (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("tile_size") = 16);
}
