#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_CONSTANT_ELEMENTS 65536  // Increased constant memory size

__constant__ float B_constant[MAX_CONSTANT_ELEMENTS];

__global__ void matmul_kernel(const float* __restrict__ A, 
                             float* __restrict__ C,
                             int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load A tile
        const int a_row = row;
        const int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K)
            As[ty][tx] = A[a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;
            
        // Load B tile
        const int b_row = t * TILE_SIZE + ty;
        const int b_col = col;
        if (b_row < K && b_col < N)
            Bs[ty][tx] = B_constant[b_row * N + b_col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    int B_elements = K * N;
    
    TORCH_CHECK(B_elements <= MAX_CONSTANT_ELEMENTS, 
              "B matrix size exceeds constant memory capacity");
    
    // Copy B to constant memory
    cudaMemcpyToSymbol(B_constant, B.data_ptr<float>(), B_elements * sizeof(float));
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory matrix multiplication (CUDA)");
}
