#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

__global__ void bmm_warp_shuffle_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Calculate warp and lane IDs
    int warp_id = threadIdx.y / warpSize;
    int lane_id = threadIdx.x % warpSize;
    
    // Shared memory for partial results
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    const unsigned FULL_MASK = 0xffffffff;
    
    // Base pointers for current batch
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_idx = t * TILE_SIZE + threadIdx.x;
        
        // Load data into shared memory
        if (row < M && k_idx < K) {
            As[threadIdx.y][threadIdx.x] = batch_A[row * K + k_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (k_idx < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = batch_B[k_idx * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products using warp shuffling
        float warp_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += WARP_SIZE) {
            // Load values for current warp
            float a_val = As[threadIdx.y][k + lane_id];
            float b_val = Bs[k + lane_id][threadIdx.x];
            
            // Perform multiplication
            warp_sum += a_val * b_val;
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
            }
            
            // First thread in warp accumulates to sum
            if (lane_id == 0) {
                sum += warp_sum;
            }
        }
        
        __syncthreads();
    }
    
    // Write result if within bounds
    if (row < M && col < N) {
        if (lane_id == 0) {  // Only first thread in warp writes result
            C[b * M * N + row * N + col] = sum;
        }
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE, 
              batch_size);

    bmm_warp_shuffle_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with warp shuffling (CUDA)");
}