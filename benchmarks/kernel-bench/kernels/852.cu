#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Kernel using shared memory for tile loading and warp-level primitives for reduction
// Block configuration: each block has (numWarpsX*32) x (numWarpsY) threads
// Each warp computes one output element C[row, col] by computing dot(A[row,*], B[*,col]).

// We set numWarpsX and numWarpsY to partition the block among warps.
// For this implementation, we choose numWarpsX = 2 and numWarpsY = 4, so each block has 2*32 = 64 threads in x and 4 threads in y, total 64*4 = 256 threads.
// Each warp processes TILE_K (set to 32) elements of the K dimension at a time using shared memory to stage the data.

#define TILE_K 32

// Kernel: each warp computes one element of the output matrix
__global__ void warp_shared_reduction_matmul_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int K, int N) {
    // Define warp partition parameters
    const int numWarpsX = 2;  // number of warps in x-direction per block
    const int numWarpsY = 4;  // number of warps in y-direction per block
    const int numWarpsPerBlock = numWarpsX * numWarpsY;  // total warps per block

    // Each warp consists of 32 threads, so blockDim.x should be numWarpsX*32.
    // Each thread's lane within its warp
    int lane = threadIdx.x % 32; 
    // Warp id in x direction
    int warp_id_x = threadIdx.x / 32;  // ranges from 0 to (numWarpsX - 1)
    // Warp id in y direction is given by threadIdx.y (range 0 to numWarpsY-1)
    int warp_id_y = threadIdx.y;
    // Global warp index within the block
    int warpIdInBlock = warp_id_y * numWarpsX + warp_id_x;  // 0 <= warpIdInBlock < numWarpsPerBlock

    // Map each warp to one output element C[row, col]
    int row = blockIdx.y * numWarpsY + warp_id_y;
    int col = blockIdx.x * numWarpsX + warp_id_x;

    float sum = 0.0f;

    // Allocate shared memory for staging tile data for each warp
    // Each warp gets TILE_K elements for A and TILE_K for B
    // Total shared memory size per block = numWarpsPerBlock * TILE_K * 2 floats
    extern __shared__ float shared_mem[];  // shared memory provided at kernel launch
    float* sA = shared_mem;  // size: numWarpsPerBlock * TILE_K
    float* sB = shared_mem + numWarpsPerBlock * TILE_K;  // size: numWarpsPerBlock * TILE_K

    // Loop over K in chunks of TILE_K
    for (int t = 0; t < K; t += TILE_K) {
        // For each tile iteration, each warp loads one element per lane into its private tile region in shared memory.
        int k_index = t + lane;  // each lane loads one element

        float a_val = 0.0f;
        float b_val = 0.0f;
        if (row < M && k_index < K) {
            a_val = A[row * K + k_index];
        }
        if (col < N && k_index < K) {
            b_val = B[k_index * N + col];
        }

        // Each warp uses its own section in shared memory
        int offset = warpIdInBlock * TILE_K + lane;
        sA[offset] = a_val;
        sB[offset] = b_val;

        // Synchronize threads within the warp to ensure shared memory is populated
        __syncwarp(0xFFFFFFFF);

        // Each warp now has a tile of size TILE_K loaded in shared memory.
        // Compute the dot product for this tile: sum over i = 0 ... TILE_K-1 of sA[warp offset + i] * sB[warp offset + i]
        // We let each thread in the warp take one element: its own lane's element, then do a warp-level reduction.
        float tile_product = sA[offset] * sB[offset];
        
        // Warp-level reduction using __shfl_down_sync
        unsigned int mask = 0xFFFFFFFF;
        for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) {
            tile_product += __shfl_down_sync(mask, tile_product, offset_shfl);
        }
        
        // Now, lane 0 of the warp holds the sum for this tile
        if (lane == 0) {
            sum += tile_product;
        }
        
        // Synchronize warp lanes before next tile iteration (not strictly necessary as each warp works independently)
        __syncwarp(0xFFFFFFFF);
    }

    // Write the final result from lane 0 of each warp to matrix C
    if (lane == 0 && row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function for launching the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    // Define block configuration
    const int numWarpsX = 2; // each block covers 2 output columns
    const int numWarpsY = 4; // each block covers 4 output rows
    dim3 block(numWarpsX * 32, numWarpsY); // blockDim.x = 2*32 = 64, blockDim.y = 4 (total 256 threads per block)

    // Grid dimensions: each block computes numWarpsX columns and numWarpsY rows of output
    dim3 grid((N + numWarpsX - 1) / numWarpsX, (M + numWarpsY - 1) / numWarpsY);

    // Shared memory size: 2 arrays of size (numWarpsPerBlock * TILE_K)
    int numWarpsPerBlock = numWarpsX * numWarpsY;
    size_t shared_memory_size = numWarpsPerBlock * TILE_K * 2 * sizeof(float);

    warp_shared_reduction_matmul_kernel<<<grid, block, shared_memory_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp shared memory reduction matrix multiplication (CUDA)");
}
