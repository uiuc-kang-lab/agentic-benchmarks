#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__global__ void einsum_kernel_strided(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,  // Total number of (batch * I * J) groups
    int L,
    int K
) {
    // Calculate base indices
    int tid = threadIdx.x;
    int p = blockIdx.x;
    
    // Each thread processes multiple k values with stride
    int k_base = blockIdx.y * (BLOCK_SIZE * ITEMS_PER_THREAD) + tid;
    
    // Only process if we have a valid p index
    if (p < IJP) {
        int offsetA = p * L;  // Base offset for A
        int offsetC = p * K;  // Base offset for C
        
        // Process multiple k values per thread
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int k = k_base + i * BLOCK_SIZE;
            if (k < K) {  // Boundary check for K dimension
                float sum = 0.0f;
                
                // Compute dot product for this k
                #pragma unroll 4
                for (int l = 0; l < L; l++) {
                    float a_val = __ldg(&A[offsetA + l]);
                    float b_val = __ldg(&B[l * K + k]);
                    sum += a_val * b_val;
                }
                
                // Store result
                C[offsetC + k] = sum;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    // Flatten the (BATCH, I, J) dimensions
    int IJP = BATCH * I * J;
    
    // Calculate grid dimensions
    int k_blocks = (K + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    dim3 blocks(IJP, k_blocks);
    dim3 threads(BLOCK_SIZE);

    einsum_kernel_strided<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with strided processing (CUDA)");
}