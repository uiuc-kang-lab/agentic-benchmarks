#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16

__global__ void einsum_kernel_optimized_streamed(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH_chunk, int I, int J, int L, int K
) {
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int b = by / (I * J);
    int i = (by % (I * J)) / J;
    int j = by % J;
    int k = bx * TILE_SIZE + tx;
    
    if (b >= BATCH_chunk || i >= I || j >= J || k >= K) return;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (L + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (tile * TILE_SIZE + ty < L && k < K) {
            B_shared[ty][tx] = B[(tile * TILE_SIZE + ty) * K + k];
        } else {
            B_shared[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int l = 0; l < TILE_SIZE && tile * TILE_SIZE + l < L; ++l) {
            int a_offset = b * I * J * L + i * J * L + j * L + tile * TILE_SIZE + l;
            sum += A[a_offset] * B_shared[l][tx];
        }
        __syncthreads();
    }
    
    if (k < K) {
        C[b * I * J * K + i * J * K + j * K + k] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    int num_streams = 4;
    int batch_chunk = (BATCH + num_streams - 1) / num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    for (int s = 0; s < num_streams; s++) {
        int start_batch = s * batch_chunk;
        int current_batch = std::min(batch_chunk, BATCH - start_batch);
        if (current_batch <= 0) break;

        const float* A_ptr = A.data_ptr<float>() + start_batch * I * J * L;
        float* C_ptr = C.data_ptr<float>() + start_batch * I * J * K;

        dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE, current_batch * I * J);

        einsum_kernel_optimized_streamed<<<blocks, threads, 0, streams[s]>>>(
            A_ptr, B.data_ptr<float>(), C_ptr,
            current_batch, I, J, L, K
        );
    }

    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}