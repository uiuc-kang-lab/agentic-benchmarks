#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Use 2D block structure for better work distribution
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Each block handles a tile of the output
    const int tile_size_x = blockDim.x;
    const int tile_size_y = blockDim.y;
    
    // Calculate starting indices for this block
    const int i_start = by * tile_size_y;
    const int j_start = bx * tile_size_x;
    
    // Calculate actual i,j coordinates for this thread
    const int i = i_start + ty;
    const int j = j_start + tx;
    
    // Process multiple batches and K values per thread if needed
    for(int b = 0; b < BATCH; ++b) {
        for(int k = 0; k < K; ++k) {
            if(i < I && j < J) {
                float sum = 0.0f;
                #pragma unroll 4
                for(int l = 0; l < L; ++l) {
                    int a_offset = b * I*J*L + i*J*L + j*L + l;
                    int b_offset = l*K + k;
                    sum += A[a_offset] * B[b_offset];
                }
                C[b * I*J*K + i*J*K + j*K + k] = sum;
            }
        }
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
    
    // Use 2D block structure
    dim3 threads(16, 16);
    dim3 blocks(
        (J + threads.x - 1) / threads.x,
        (I + threads.y - 1) / threads.y
    );
    
    einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}