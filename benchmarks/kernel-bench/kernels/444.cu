#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t start_row,
    int64_t K,
    int64_t M)
{
    __shared__ scalar_t warp_sums[WARPS_PER_BLOCK] __attribute__((aligned(16)));
    
    const int64_t row = start_row + blockIdx.x;
    if (row >= M) return;
    
    const int64_t tid = threadIdx.x;
    const int64_t warp_id = tid / WARP_SIZE;
    const int64_t lane = tid % WARP_SIZE;
    
    scalar_t sum = 0;
    const scalar_t* row_ptr = A + row * K;
    
#pragma unroll 4
    for (int64_t k = tid; k < K; k += BLOCK_SIZE) {
        sum += row_ptr[k] * B[k];
    }
    
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        sum = (tid < WARPS_PER_BLOCK) ? warp_sums[tid] : 0;
        
#pragma unroll
        for (int offset = (WARPS_PER_BLOCK + 1)/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            C[row] = sum;
        }
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous().view({-1});
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    auto C = torch::empty({M}, A.options());
    
    constexpr int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int64_t chunk_size = (M + num_streams - 1) / num_streams;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        scalar_t* C_ptr = C.data_ptr<scalar_t>();
        
        for (int i = 0; i < num_streams; ++i) {
            const int64_t start = i * chunk_size;
            const int64_t end = (i == num_streams-1) ? M : (i+1)*chunk_size;
            const int64_t rows_in_chunk = end - start;
            
            if (rows_in_chunk <= 0) continue;
            
            // Async initialization for this chunk
            cudaMemsetAsync(C_ptr + start, 0, rows_in_chunk * sizeof(scalar_t), streams[i]);
            
            dim3 threads(BLOCK_SIZE);
            dim3 blocks(rows_in_chunk);
            
            matvec_mul_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                A_contig.data_ptr<scalar_t>(),
                B_contig.data_ptr<scalar_t>(),
                C_ptr,
                start,
                K,
                M
            );
        }
    }));
    
    // Sync and cleanup streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}
