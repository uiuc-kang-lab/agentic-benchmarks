#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 8
#define TILE_DIM 16

template <typename scalar_t>
__global__ void warp_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t shared_mem[BLOCK_ROWS][BLOCK_COLS];
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    const int col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    
    scalar_t thread_sum = 0;
    
    if (row < N * M && col < L) {
        const int batch_idx = row / M;
        const int m_idx = row % M;
        
        for (int t = 0; t < K; t += TILE_DIM) {
            __shared__ scalar_t As[TILE_DIM][TILE_DIM];
            __shared__ scalar_t Bs[TILE_DIM][TILE_DIM];
            
            if (threadIdx.x < TILE_DIM && t + threadIdx.y < K) {
                As[threadIdx.y][threadIdx.x] = A[batch_idx * M * K + m_idx * K + t + threadIdx.y];
            }
            if (threadIdx.y < TILE_DIM && t + threadIdx.x < K) {
                Bs[threadIdx.x][threadIdx.y] = B[(t + threadIdx.x) * L + col];
            }
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                if (t + k < K) {
                    thread_sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    if (lane == 0) {
        shared_mem[threadIdx.y][threadIdx.x / WARP_SIZE] = thread_sum;
    }
    
    __syncthreads();
    
    if (wid == 0) {
        thread_sum = (lane < (blockDim.x * blockDim.y + WARP_SIZE - 1) / WARP_SIZE) 
            ? shared_mem[threadIdx.y][lane] : 0;
            
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (lane == 0 && row < N * M && col < L) {
            output[row * L + col] = thread_sum;
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);
    
    dim3 threads(BLOCK_COLS, BLOCK_ROWS);
    dim3 grid((L + BLOCK_COLS - 1) / BLOCK_COLS, 
              (N * M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        warp_optimized_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);
    
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Warp-optimized tensor-matrix multiplication (CUDA)");
}