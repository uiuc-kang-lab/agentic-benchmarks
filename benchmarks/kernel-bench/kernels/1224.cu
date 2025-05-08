#include <torch/extension.h>
#include <cuda_runtime.h>

// Device function to load a vector from global memory into shared memory
__device__ inline void load_shared_A(const float* __restrict__ A, float* shared_A, int L) {
    int tid = threadIdx.x;
    for (int l = tid; l < L; l += blockDim.x) {
        shared_A[l] = A[l];
    }
}

// Device function to compute the dot product between shared_A and a column of B
// B is stored in row-major order with dimensions [L x K]
__device__ inline float compute_dot(const float* shared_A, const float* __restrict__ B, int L, int K, int k) {
    float sum = 0.0f;
    int l = 0;
    #pragma unroll 4
    for (; l + 3 < L; l += 4) {
        sum += shared_A[l]     * B[l * K + k] +
               shared_A[l + 1] * B[(l + 1) * K + k] +
               shared_A[l + 2] * B[(l + 2) * K + k] +
               shared_A[l + 3] * B[(l + 3) * K + k];
    }
    for (; l < L; ++l) {
        sum += shared_A[l] * B[l * K + k];
    }
    return sum;
}

// Kernel computing C = einsum('bijl,lk->bijk') by processing one (b,i,j) triplet per block
__global__ void einsum_kernel_modular(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K) {

    // Determine the (b, i, j) indices for this block
    int idx = blockIdx.x;
    int b = idx / (I * J);
    int rem = idx % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Offsets for A and C for the (b,i,j) slice
    int a_offset = b * (I * J * L) + i * (J * L) + j * L;
    int c_offset = b * (I * J * K) + i * (J * K) + j * K;

    // Allocate shared memory to store A[b,i,j,*]
    extern __shared__ float shared_A[];  // size = L floats
    load_shared_A(&A[a_offset], shared_A, L);
    __syncthreads();

    // Each thread computes one or more k indices in C
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float sum = compute_dot(shared_A, B, L, K, k);
        C[c_offset + k] = sum;
    }
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
    
    // One block per (b, i, j) slice
    int numBlocks = BATCH * I * J;
    int threads = 256;
    size_t sharedMemBytes = L * sizeof(float);

    einsum_kernel_modular<<<numBlocks, threads, sharedMemBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Einsum kernel with modular device functions (CUDA)");
}
