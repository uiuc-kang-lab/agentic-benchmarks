#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4
#define TILE_SIZE 32  // Size of shared memory tile

// Kernel with shared memory and CUDA streams
__global__ void einsum_kernel_streamed(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,  // Total number of groups (BATCH * I * J)
    int L,
    int K
) {
    int p = blockIdx.x;
    int tile_idx = blockIdx.y;
    int tid = threadIdx.x;

    int k_base = tile_idx * (BLOCK_SIZE * ITEMS_PER_THREAD) + tid;
    int offsetA = p * L;
    int offsetC = p * K;

    // Shared memory for A and B tiles
    __shared__ float shared_A[TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][BLOCK_SIZE];

    float sums[ITEMS_PER_THREAD] = {0.0f};

    // Process input in tiles
    for (int l_base = 0; l_base < L; l_base += TILE_SIZE) {
        // Load A tile into shared memory
        if (tid < min(TILE_SIZE, L - l_base)) {
            shared_A[tid] = A[offsetA + l_base + tid];
        }

        // Load B tile into shared memory
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int k = k_base + i * BLOCK_SIZE;
            if (k < K && tid < TILE_SIZE) {
                for (int t = 0; t < min(TILE_SIZE, L - l_base); t++) {
                    shared_B[t][tid] = B[(l_base + t) * K + k];
                }
            }
        }
        __syncthreads();

        // Compute partial sums using shared memory
        #pragma unroll
        for (int l_offset = 0; l_offset < min(TILE_SIZE, L - l_base); l_offset++) {
            float a_val = shared_A[l_offset];
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                int k = k_base + i * BLOCK_SIZE;
                if (k < K) {
                    sums[i] += a_val * shared_B[l_offset][tid];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int k = k_base + i * BLOCK_SIZE;
        if (k < K) {
            C[offsetC + k] = sums[i];
        }
    }
}

// Forward function with CUDA streams
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    int IJP = BATCH * I * J;
    int k_tiles = (K + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    dim3 blocks(IJP, k_tiles);
    dim3 threads(BLOCK_SIZE);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernel in stream1
    einsum_kernel_streamed<<<blocks, threads, 0, stream1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 4D tensor-matrix multiplication with CUDA streams (CUDA)");
}