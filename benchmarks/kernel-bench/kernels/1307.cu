#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    __shared__ float warp_sums[TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int lane_id = tx % WARP_SIZE;
    int warp_id = ty;
    
    int batch_i = blockIdx.z / I;
    int idx_i = blockIdx.z % I;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.x * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // Process L dimension in chunks of WARP_SIZE
    for (int l_base = 0; l_base < L; l_base += WARP_SIZE) {
        int l = l_base + lane_id;
        
        float a_val = (j < J && l < L) ? 
            __ldg(&A[batch_i * I*J*L + idx_i * J*L + j * L + l]) : 0.0f;
        float b_val = (l < L && k < K) ? 
            __ldg(&B[l * K + k]) : 0.0f;
            
        float prod = a_val * b_val;
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }
        
        // First thread in warp stores the result
        if (lane_id == 0) {
            sum += prod;
        }
    }
    
    // Store warp results to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction: accumulate partial sums from each warp in shared memory
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && j < J && k < K) {
        float final_sum = 0.0f;
        // Accumulate results stored by each warp
        for (int i = 0; i < blockDim.y; i++) {
            final_sum += warp_sums[i];
        }
        C[batch_i * I*J*K + idx_i * J*K + j * K + k] = final_sum;
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
    
    dim3 threads(WARP_SIZE, TILE_SIZE/WARP_SIZE);
    dim3 blocks(
        (K + WARP_SIZE - 1) / WARP_SIZE,
        (J + TILE_SIZE - 1) / TILE_SIZE,
        BATCH * I
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
    m.def("forward", &forward, "4D tensor-matrix multiplication with warp-level optimizations (CUDA)");
}