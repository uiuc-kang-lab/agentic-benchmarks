#include <torch/extension.h>
#include <cuda_runtime.h>

// Device function to load A into shared memory
__device__ void load_a_to_shared(
    const float* __restrict__ A,
    float* shared_A,
    int b, int i, int j,
    int I, int J, int L
) {
    int tid = threadIdx.x;
    for (int l = tid; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        shared_A[l] = A[a_index];
    }
    __syncthreads();
}

// Device function to compute dot product
__device__ float compute_dot_product(
    const float* shared_A,
    const float* __restrict__ B,
    int k, int K, int L
) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int l = 0; l < L; l++) {
        sum += shared_A[l] * B[l * K + k];
    }
    return sum;
}

// Device function to store result
__device__ void store_result(
    float* __restrict__ C,
    float result,
    int b, int i, int j, int k,
    int I, int J, int K
) {
    int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
    C[c_index] = result;
}

__global__ void einsum_kernel_modular(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Compute indices for b, i, j based on blockIdx
    int index = blockIdx.x;
    int b = index / (I * J);
    int rem = index % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Allocate shared memory for the entire vector A[b,i,j,*]
    extern __shared__ float shared_A[];

    // Load A[b,i,j,*] into shared memory
    load_a_to_shared(A, shared_A, b, i, j, I, J, L);

    // Each thread computes one or more elements in the K dimension
    int tid = threadIdx.x;
    for (int k = tid; k < K; k += blockDim.x) {
        // Compute dot product
        float result = compute_dot_product(shared_A, B, k, K, L);
        
        // Store the result
        store_result(C, result, b, i, j, k, I, J, K);
    }
}

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

    // Each block processes one (b, i, j) triple
    int numBlocks = BATCH * I * J;
    int threads = 256;

    // Shared memory to hold A[b,i,j,*]
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
    m.def("forward", &forward, "Modular 4D tensor-matrix multiplication (CUDA)");
}