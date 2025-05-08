#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)
#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t start_row,
    int64_t K,
    int64_t M)
{
    const int64_t row = start_row + blockIdx.x;
    if(row >= M) return;
    
    const int64_t tid = threadIdx.x;
    const int64_t warp_id = tid / WARP_SIZE;
    const int64_t lane = tid % WARP_SIZE;
    
    scalar_t sum = 0;
    const scalar_t* row_ptr = A + row * K;
    
    #pragma unroll 4
    for(int64_t k = tid; k < K; k += BLOCK_SIZE) {
        sum += __ldg(&row_ptr[k]) * __ldg(&B[k]);
    }
    
    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if(lane == 0) {
        atomicAdd(&C[row], sum);
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous().view({-1});
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    auto C = torch::zeros({M}, A.options());
    
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    const int64_t chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        scalar_t* C_ptr = C.data_ptr<scalar_t>();
        
        for(int i = 0; i < NUM_STREAMS; ++i) {
            const int64_t start = i * chunk_size;
            const int64_t end = (i == NUM_STREAMS-1) ? M : (i+1)*chunk_size;
            const int64_t rows_in_chunk = end - start;
            
            if(rows_in_chunk <= 0) continue;
            
            // Asynchronous initialization for this chunk
            cudaMemsetAsync(C_ptr + start, 0, rows_in_chunk*sizeof(scalar_t), streams[i]);
            
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
    
    for(int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}