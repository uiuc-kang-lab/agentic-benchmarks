#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Define tile dimensions for warp-collaborative multiplication
// BM: number of output rows computed per block (per block tile)
// BN: number of output columns computed per block
// KT: tile width along K dimension

#define BM 4
#define BN 8
#define KT 32

// Kernel: each warp computes one element of matrix C using warp-level reduction (using __shfl_down_sync).
// Block configuration: blockDim.x = 32 (warp size), blockDim.y = BM * BN (i.e. 32 threads => 32 warps per block)
// Each warp is identified by its index (from 0 to BM*BN-1) and computes one output element

__global__ void matmul_warp_reduce_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    // Each block computes a tile of C of size BM x BN
    // There are BM*BN warps per block; each warp computes one output element
    // Within each warp, threads are identified by lane (0-31)

    int lane = threadIdx.x;  // 0..31
    int warp_id = threadIdx.y;  // Each row of block (of size 32 threads) is one warp

    // Map warp_id to output tile coordinates (within the block tile)
    int warp_row = warp_id / BN;   // warp_row in [0, BM)
    int warp_col = warp_id % BN;   // warp_col in [0, BN)

    // Global output indices
    int row = blockIdx.y * BM + warp_row;
    int col = blockIdx.x * BN + warp_col;

    float sum = 0.0f;

    // Number of k-tiles
    int numTiles = (N + KT - 1) / KT;

    // Shared memory: we load a tile from A and a tile from B for reuse
    // sA: BM x KT tile from A
    // sB: KT x BN tile from B
    extern __shared__ float shared[]; // total size: BM*KT + KT*BN floats
    float* sA = shared;               // sA[0 .. BM*KT - 1]
    float* sB = sA + BM * KT;           // sB[0 .. KT*BN - 1]

    // Loop over tiles in the K dimension
    for (int m = 0; m < numTiles; m++) {
        // Cooperative loading of A and B tiles into shared memory
        // Total elements to load: A_tile: BM * KT, B_tile: KT * BN
        int totalA = BM * KT;
        int totalB = KT * BN;
        int tid = threadIdx.y * blockDim.x + threadIdx.x; // linear thread id in block
        int blockSize = blockDim.x * blockDim.y;

        // Load A tile: from A[blockIdx.y*BM ... blockIdx.y*BM+BM-1][m*KT ... m*KT+KT-1]
        for (int i = tid; i < totalA; i += blockSize) {
            int a_row = i / KT;
            int a_col = i % KT;
            int global_row = blockIdx.y * BM + a_row;
            int global_col = m * KT + a_col;
            if (global_row < N && global_col < N)
                sA[i] = A[global_row * N + global_col];
            else
                sA[i] = 0.0f;
        }

        // Load B tile: from B[m*KT ... m*KT+KT-1][blockIdx.x*BN ... blockIdx.x*BN+BN-1]
        for (int j = tid; j < totalB; j += blockSize) {
            int b_row = j / BN;
            int b_col = j % BN;
            int global_row = m * KT + b_row;
            int global_col = blockIdx.x * BN + b_col;
            if (global_row < N && global_col < N)
                sB[j] = B[global_row * N + global_col];
            else
                sB[j] = 0.0f;
        }
        __syncthreads();

        // Each warp computes a partial dot product for its assigned output element.
        // Dot product: for k = 0 .. KT-1, accumulate sA[warp_row][k] * sB[k][warp_col]
        // Partition the work among the 32 lanes: each lane handles one k (since KT == 32) 
        float partial = 0.0f;
        // Since lane is in [0, 31] and KT == 32, each lane processes one element
        {
            float a_val = sA[warp_row * KT + lane];
            float b_val = sB[lane * BN + warp_col];
            partial = a_val * b_val;
        }
        
        // Use warp-level reduction to sum the partial results
        for (int offset = 16; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }

        // Lane 0 of each warp adds the result of this tile to the final sum
        if (lane == 0) {
            sum += partial;
        }
        __syncthreads();
    }

    // Write the computed output element to global memory (only lane 0 of each warp does this)
    if (lane == 0 && row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function that wraps the kernel launch

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    // Grid dimensions based on output tile size
    // Each block computes a BM x BN tile of C
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    // Block dimensions: we want one warp per output element in the tile, so total warps = BM*BN
    // Each warp has 32 threads, so blockDim.x = 32, blockDim.y = BM*BN
    dim3 block(32, BM * BN);

    // Shared memory size: (BM*KT + KT*BN) floats
    size_t sharedMemSize = (BM * KT + KT * BN) * sizeof(float);

    matmul_warp_reduce_kernel<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel with warp-level reduction (CUDA)");
}
