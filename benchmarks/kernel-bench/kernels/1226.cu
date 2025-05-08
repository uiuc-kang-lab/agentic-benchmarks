#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Kernel using shared memory and warp-level primitives for reduction
__global__ void einsum_kernel_warp_reduce(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Each block computes output for a single (b, i, j) triple.
    // gridDim.x indexes (b, i, j) and gridDim.y partitions the K dimension.
    int index = blockIdx.x;
    int b = index / (I * J);
    int rem = index % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Determine k based on gridDim.y and threadIdx.y
    int k_base = blockIdx.y * blockDim.y;
    int k = k_base + threadIdx.y;

    // Allocate shared memory for A[b, i, j, :]
    extern __shared__ float sA[];  // Size: L * sizeof(float)

    // Only one row (threadIdx.y == 0) loads the vector to shared memory
    if (threadIdx.y == 0) {
        for (int l = threadIdx.x; l < L; l += blockDim.x) {
            int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
            sA[l] = A[a_index];
        }
    }
    __syncthreads();

    float partial = 0.0f;
    if (k < K) {
        // Each thread in the warp processes a subset of the L dimension
        for (int l = threadIdx.x; l < L; l += blockDim.x) {
            partial += sA[l] * B[l * K + k];
        }
    }

    // Perform warp-level reduction across threadIdx.x using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(mask, partial, offset);
    }

    // Only lane 0 in each warp writes the result
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0 && (k < K)) {
        int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
        C[c_index] = partial;
    }
}

// Forward function launches the kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "Tensor A must be 4D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Grid dims:
    //  - grid.x corresponds to each (b, i, j) triple
    //  - grid.y partitions the K dimension
    int numBlocks_x = BATCH * I * J;
    int blockDim_x = WARP_SIZE;     // Use one warp for reduction
    int blockDim_y = 8;             // Number of k elements processed per block along y
    int grid_y = (K + blockDim_y - 1) / blockDim_y;

    dim3 grid(numBlocks_x, grid_y);
    dim3 block(blockDim_x, blockDim_y);

    size_t sharedMemBytes = L * sizeof(float);

    einsum_kernel_warp_reduce<<<grid, block, sharedMemBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with warp-level shared memory reduction (CUDA)");
}
