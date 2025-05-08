#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void strided_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    // Grid-stride loop for row and column indices
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < N; row += blockDim.y * gridDim.y) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col <= row; col += blockDim.x * gridDim.x) {
            float sum = 0.0f;

            // Precompute loop boundaries
            const int k_start = col;
            const int k_end = row + 1;
            const int num_tiles = (k_end - k_start + TILE_SIZE - 1) / TILE_SIZE;

            for (int t = 0; t < num_tiles; ++t) {
                const int tile_start = k_start + t * TILE_SIZE;
                const int tile_end = min(tile_start + TILE_SIZE, k_end);

                // Load tiles with coalesced accesses
                const int load_row = tile_start + threadIdx.y;
                if (load_row <= row && tile_start + threadIdx.x < N) {
                    As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tile_start + threadIdx.x]);
                    Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(tile_start + threadIdx.y) * N + col]);
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
                __syncthreads();

                // Compute tile contribution
                const int compute_start = max(tile_start, k_start);
                const int compute_end = min(tile_end, k_end);
                for (int k = compute_start; k < compute_end; ++k) {
                    const int k_offset = k - tile_start;
                    sum += As[threadIdx.y][k_offset] * Bs[k_offset][threadIdx.x];
                }
                __syncthreads();
            }

            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Configure kernel to use 80% of available SMs
    const int device_id = A.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    const int max_blocks = 0.8 * prop.multiProcessorCount;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(min(max_blocks, (N + TILE_SIZE - 1) / TILE_SIZE),
             min(max_blocks, (N + TILE_SIZE - 1) / TILE_SIZE));

    strided_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                 B.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided lower triangular matrix multiplication (CUDA)");
}