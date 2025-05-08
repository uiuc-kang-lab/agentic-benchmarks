#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void triangular_mm_shared_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1)/TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        int load_row = row;
        int load_col = tile * TILE_SIZE + threadIdx.x;
        if (load_col > tile * TILE_SIZE + threadIdx.y) continue;
        
        if (load_row < N && load_col <= load_row)
            sA[threadIdx.y][threadIdx.x] = A[load_row * N + load_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        
        load_row = tile * TILE_SIZE + threadIdx.y;
        load_col = col;
        if (load_row < N && load_col <= load_row)
            sB[threadIdx.y][threadIdx.x] = B[load_row * N + load_col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < N && col < N && row >= col)
        C[row * N + col] = sum;
    else if (row < N && col < N)
        C[row * N + col] = 0.f;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    
    auto C = torch::zeros_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE);
    
    triangular_mm_shared_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular MM with shared memory");
}