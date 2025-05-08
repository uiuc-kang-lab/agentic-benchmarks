#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel with 2D grid indexing: gridIdx.x tiles the K dimension and gridIdx.y selects the (b,i,j) slice
__global__ void einsum_kernel_tiled_2d(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Each block handles one tile in the K dimension for a fixed (b, i, j)
    int tile_k = blockDim.x;  // tile size equals the number of threads in x-dimension
    int k = blockIdx.x * tile_k + threadIdx.x;

    // gridIdx.y encodes the (b, i, j) indices (flattened)
    int idx = blockIdx.y;
    int b = idx / (I * J);
    int rem = idx % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Boundary check for K dimension
    if (k >= K) return;
    
    // Allocate shared memory for A[b,i,j,*]
    extern __shared__ float shared_A[]; // Shared memory size should be L floats
    
    // Cooperatively load the entire A[b,i,j,*] vector into shared memory
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        shared_A[l] = A[a_index];
    }
    __syncthreads();

    // Compute the dot product for the output element C[b,i,j,k]
    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        sum += shared_A[l] * B[l * K + k];
    }

    int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
    C[c_index] = sum;
}

// Forward function to launch the kernel

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

    // Configure 2D grid: 
    //  - grid.x divides the K dimension into tiles of size blockDim.x
    //  - grid.y assigns one block per (b, i, j) slice
    int tile_k = 256;  // Number of threads in x-dimension
    dim3 block(tile_k);
    dim3 grid((K + tile_k - 1) / tile_k, BATCH * I * J);

    size_t sharedMemSize = L * sizeof(float);

    einsum_kernel_tiled_2d<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with 2D tiled indexing (CUDA)");
}
