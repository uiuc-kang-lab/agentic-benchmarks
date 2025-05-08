#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Early exit for upper triangular part
    if (row < col) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    
    // Loop over tiles
    const int start_tile = col/TILE_SIZE;
    const int end_tile = row/TILE_SIZE;
    
    for (int t = start_tile; t <= end_tile; t++) {
        // Prefetch next tile (could be implemented for further optimization)
        
        // Collaborative loading with coalesced memory access
        if (row < N && (t*TILE_SIZE + tx) <= row) {
            As[ty][tx] = A[row * N + (t*TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t*TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(t*TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile using vectorized operations
        if (row < N && col < N) {
            float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k += 4) {
                if ((t*TILE_SIZE + k) >= col && (t*TILE_SIZE + k) <= row) {
                    float4 a_vec = *reinterpret_cast<const float4*>(&As[ty][k]);
                    float4 b_vec = *reinterpret_cast<const float4*>(&Bs[k][tx]);
                    sum_vec.x += a_vec.x * b_vec.x;
                    sum_vec.y += a_vec.y * b_vec.y;
                    sum_vec.z += a_vec.z * b_vec.z;
                    sum_vec.w += a_vec.w * b_vec.w;
                }
            }
            sum += sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Inputs must have same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    // Use multiple CUDA streams for potential overlap
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}