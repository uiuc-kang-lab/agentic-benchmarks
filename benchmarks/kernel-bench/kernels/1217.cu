#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel_hybrid(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Get b,i,j indices from block
    int block_idx = blockIdx.x;
    int b = block_idx / (I * J);
    int remainder = block_idx % (I * J);
    int i = remainder / J;
    int j = remainder % J;

    // Shared memory for A[b,i,j,:]
    extern __shared__ float shared_A[];
    
    // Cooperatively load A[b,i,j,:] into shared memory
    int tid = threadIdx.x;
    for (int l = tid; l < L; l += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
        shared_A[l] = A[a_index];
    }
    __syncthreads();

    // Each thread handles multiple k values with unrolled inner loop
    for (int k = tid; k < K; k += blockDim.x) {
        float sum = 0.0f;
        
        // Unrolled inner loop for L dimension
        int l = 0;
        #pragma unroll 4
        for (; l + 3 < L; l += 4) {
            sum += shared_A[l] * B[l * K + k] +
                   shared_A[l+1] * B[(l+1) * K + k] +
                   shared_A[l+2] * B[(l+2) * K + k] +
                   shared_A[l+3] * B[(l+3) * K + k];
        }
        
        // Handle remaining elements
        for (; l < L; l++) {
            sum += shared_A[l] * B[l * K + k];
        }
        
        // Store result
        int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
        C[c_index] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    int numBlocks = BATCH * I * J;
    int threads = 256;
    size_t sharedMemSize = L * sizeof(float);
    
    einsum_kernel_hybrid<<<numBlocks, threads, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid optimized 4D tensor-matrix multiplication (CUDA)");
}