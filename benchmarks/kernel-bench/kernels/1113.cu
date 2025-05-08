#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {
    
    // Use shared memory for tiles of A and B
    __shared__ scalar_t As[32][32];
    __shared__ scalar_t Bs[32][32];
    
    const int TILE_SIZE = 32;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Register to accumulate results
    scalar_t sum = 0;
    
    // Calculate batch index
    const int batch = blockIdx.z;
    
    if (batch < N && row < M && col < L) {
        // Loop over tiles
        const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int t = 0; t < numTiles; ++t) {
            // Load tile into shared memory
            const int tileK = t * TILE_SIZE;
            
            if (tileK + tx < K && row < M)
                As[ty][tx] = A[batch * M * K + row * K + tileK + tx];
            else
                As[ty][tx] = 0;
                
            if (tileK + ty < K && col < L)
                Bs[ty][tx] = B[(tileK + ty) * L + col];
            else
                Bs[ty][tx] = 0;
                
            __syncthreads();
            
            // Compute on the tile
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }
        
        // Write result
        if (row < M && col < L)
            output[batch * M * L + row * L + col] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    const int threads = 1024;
    const int total_elements = N * M * L;
    const int blocks = (total_elements + threads - 1) / threads;

    // Create a CUDA stream
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Transfer inputs A and B to the device
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        scalar_t* d_A;
        scalar_t* d_B;
        scalar_t* d_output;

        cudaMalloc(&d_A, A.numel() * sizeof(scalar_t));
        cudaMalloc(&d_B, B.numel() * sizeof(scalar_t));
        cudaMalloc(&d_output, output.numel() * sizeof(scalar_t));

        cudaMemcpyAsync(d_A, A.data_ptr<scalar_t>(), A.numel() * sizeof(scalar_t), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B, B.data_ptr<scalar_t>(), B.numel() * sizeof(scalar_t), cudaMemcpyHostToDevice, stream2);

        // Launch kernel on stream1
        module_fn_cuda_kernel<scalar_t><<<blocks, threads, 0, stream1>>>(d_A, d_B, d_output, N, M, K, L);

        // Copy result from device to host
        cudaMemcpyAsync(output.data_ptr<scalar_t>(), d_output, output.numel() * sizeof(scalar_t), cudaMemcpyDeviceToHost, stream1);

        // Synchronize streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_output);
    }));

    // Ensure synchronization
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
  m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}