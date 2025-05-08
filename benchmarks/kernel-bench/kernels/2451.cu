#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for output per block
#define TILE_M 2      // number of output rows computed per block
#define TILE_N 2      // number of output columns computed per block

// Each output element is computed cooperatively by 2 warps
#define WARPS_PER_ELEMENT 2
#define THREADS_PER_WARP 32
#define THREADS_PER_ELEMENT (WARPS_PER_ELEMENT * THREADS_PER_WARP)  // 64 threads per output element

// Kernel: Compute C = A^T * B^T, where A is (K x M) and B is (N x K), so C is (M x N)
// Each output element C[m][n] = sum_{k=0}^{K-1} A[k * M + m] * B[n * K + k]
// For each output element, a group of 64 threads (2 warps) cooperatively compute the dot product.
// The K-dimension is partitioned among the 2 warps so that they work on disjoint segments. 
// Each warp performs an intra-warp reduction using __shfl_down_sync and writes its partial result to shared memory. 
// Finally, one thread per output element loads both partial sums from shared memory and writes the final sum to global memory.


template <typename scalar_t>
__global__ void warp_shared_reduce_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, // A: (K x M), so M is number of columns in A
    int N, // B: (N x K), so N is number of rows in B
    int K  // reduction dimension
) {
    // Each block computes a tile of C of size TILE_M x TILE_N.
    // Each output element in this tile is computed by a group of WARPS_PER_ELEMENT warps.
    // Block configuration: blockDim.x = 32 (one warp's width), blockDim.y = TILE_M * TILE_N * WARPS_PER_ELEMENT

    // Identify the group index (i.e. which output element in the tile) for this thread
    int warp_pair_idx = threadIdx.y / WARPS_PER_ELEMENT;  // ranges from 0 to (TILE_M*TILE_N - 1)
    int lane = threadIdx.x;  // lane within a warp [0, 31]

    // Map the group index to the final output coordinates
    int tile_start_row = blockIdx.y * TILE_M;
    int tile_start_col = blockIdx.x * TILE_N;
    int out_row = tile_start_row + (warp_pair_idx / TILE_N);
    int out_col = tile_start_col + (warp_pair_idx % TILE_N);

    // Within the group computing the same output element, determine which warp this thread belongs to
    int warp_in_pair = threadIdx.y % WARPS_PER_ELEMENT;  // 0 or 1
    // Partition the K dimension among the WARPS_PER_ELEMENT warps
    // Each warp will start at a different offset and stride by the total number of threads in the group.
    int offset = warp_in_pair * THREADS_PER_WARP; // 0 for first warp, 32 for second warp

    scalar_t sum = 0;
    // Each thread processes a subset of the K dimension with stride = THREADS_PER_ELEMENT (64)
    for (int k = lane + offset; k < K; k += THREADS_PER_ELEMENT) {
        // A is stored as (K x M): A[k, m] is at A[k * M + m]
        // B is stored as (N x K): B[n, k] is at B[n * K + k]
        scalar_t a_val = __ldg(&A[k * M + out_row]);
        scalar_t b_val = __ldg(&B[out_col * K + k]);
        sum += a_val * b_val;
    }

    // Intra-warp reduction using warp shuffle
    for (int delta = THREADS_PER_WARP / 2; delta > 0; delta /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    }

    // Use shared memory to combine the two warps' results for each output element
    extern __shared__ char smem[];
    scalar_t* shared_sums = (scalar_t*)smem;
    // Each group (output element) reserves WARPS_PER_ELEMENT slots in shared memory
    int shared_index = warp_pair_idx * WARPS_PER_ELEMENT + warp_in_pair;
    if (lane == 0) {
        shared_sums[shared_index] = sum;
    }
    __syncthreads();

    // Final reduction: only one thread per group (from the first warp) sums the two partial results
    if ((warp_in_pair == 0) && (lane == 0)) {
        scalar_t final_sum = shared_sums[warp_pair_idx * WARPS_PER_ELEMENT] + shared_sums[warp_pair_idx * WARPS_PER_ELEMENT + 1];
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = final_sum;
        }
    }
}


// PyTorch binding function
// A: (K x M), B: (N x K), so C: (M x N)
// Grid dimensions: gridDim.x = ceil(N / TILE_N), gridDim.y = ceil(M / TILE_M)
// Block dimensions: blockDim.x = THREADS_PER_WARP (32), blockDim.y = TILE_M * TILE_N * WARPS_PER_ELEMENT
// Shared memory size per block: TILE_M * TILE_N * WARPS_PER_ELEMENT * sizeof(scalar_t)

torch::Tensor warp_shared_reduce_matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(THREADS_PER_WARP, TILE_M * TILE_N * WARPS_PER_ELEMENT); // e.g., (32, 2*2*2 = 8) => 256 threads per block
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    size_t shared_mem_size = TILE_M * TILE_N * WARPS_PER_ELEMENT * sizeof(A.scalar_type().elementSize());
    // Instead of A.scalar_type().elementSize(), we want sizeof(scalar_t) but use AT_DISPATCH to be general

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "warp_shared_reduce_matmul_transpose_kernel", ([&] {
        warp_shared_reduce_matmul_transpose_kernel<scalar_t><<<blocks, threads, TILE_M * TILE_N * WARPS_PER_ELEMENT * sizeof(scalar_t)>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_shared_reduce_matmul_transpose_cuda, "Matrix multiplication with transposed inputs using shared memory and warp-level reduction (CUDA)");
}
