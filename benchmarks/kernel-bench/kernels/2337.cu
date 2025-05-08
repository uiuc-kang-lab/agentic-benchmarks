#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for output computed per block
// Each warp (32 threads) computes one output element
static const int CHUNK_SIZE = 32;
static const int TILE_M = 4;  // number of output rows per block
static const int TILE_N = 4;  // number of output cols per block

// Kernel: Each warp computes C[m][n] = dot(A[m, :], B[n, :])
// where A is (M x K) and B is (N x K). Note: This corresponds to C = A * B^T
// Mapping:
//   - Each block has blockDim.x = 32 (warp width) and blockDim.y = TILE_M * TILE_N (one warp per output element)
//   - For a given warp, warp_id = threadIdx.y, lane = threadIdx.x
//   - The output element indices are computed as:
//         m = blockIdx.y * TILE_M + (warp_id / TILE_N)
//         n = blockIdx.x * TILE_N + (warp_id % TILE_N)
//   - The dot product is computed in chunks of size CHUNK_SIZE. For each chunk, the warp cooperatively loads
//     a sub-vector of A[m, k0 : k0+CHUNK_SIZE] and B[n, k0 : k0+CHUNK_SIZE] into shared memory and then each lane
//     computes the product for its assigned index. A warp-level reduction using __shfl_down_sync() sums the partial
//     products from all 32 lanes. The per-chunk result (available from lane 0) is accumulated into the final sum.

__global__ void matmul_transposed_kernel_optimized(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int M, int N, int K) {
    // Each block computes a tile of size (TILE_M x TILE_N) output elements, one per warp.
    // Block configuration: dim3 block(32, TILE_M*TILE_N) i.e. 32 threads per warp, and one warp per output element.
    int warp_id = threadIdx.y; // 0 .. (TILE_M*TILE_N - 1)
    int lane = threadIdx.x;    // 0..31

    // Compute output element indices for this warp
    int warp_row = warp_id / TILE_N;   // which output row (within the block tile)
    int warp_col = warp_id % TILE_N;     // which output col (within the block tile)

    int m = blockIdx.y * TILE_M + warp_row;
    int n = blockIdx.x * TILE_N + warp_col;

    float sum = 0.0f;

    // Allocate shared memory for each warp's CHUNK_SIZE from A and B
    // Each warp uses 2 * CHUNK_SIZE floats in shared memory (first CHUNK_SIZE for A, next for B)
    extern __shared__ float shared[];
    float* sA = shared + warp_id * CHUNK_SIZE * 2; // first CHUNK_SIZE for A
    float* sB = sA + CHUNK_SIZE;                   // next CHUNK_SIZE for B

    // Loop over K in chunks of CHUNK_SIZE
    for (int k0 = 0; k0 < K; k0 += CHUNK_SIZE) {
        int current_chunk = (k0 + CHUNK_SIZE <= K) ? CHUNK_SIZE : (K - k0);

        // Each lane loads one element of the current chunk into shared memory if within bounds
        if (m < M && (k0 + lane) < K) {
            sA[lane] = A[m * K + k0 + lane];
        } else if(lane < current_chunk) {
            sA[lane] = 0.0f;
        }
        if (n < N && (k0 + lane) < K) {
            sB[lane] = B[n * K + k0 + lane];
        } else if(lane < current_chunk) {
            sB[lane] = 0.0f;
        }

        // Ensure all lanes in the warp have loaded their data
        __syncwarp();

        // Each lane computes product for its assigned element if within the current chunk size
        float temp = 0.0f;
        if (lane < current_chunk) {
            temp = sA[lane] * sB[lane];
        }

        // Warp-level reduction using __shfl_down_sync to sum the 32 values
        for (int offset = 16; offset > 0; offset /= 2) {
            temp += __shfl_down_sync(0xffffffff, temp, offset);
        }

        // Lane 0 of the warp adds the reduced sum for this chunk to the overall sum
        if (lane == 0) {
            sum += temp;
        }

        // Ensure warp sync before next chunk
        __syncwarp();
    }

    // Write the result to C if within bounds; only lane 0 writes the final computed value
    if (m < M && n < N && lane == 0) {
        C[m * N + n] = sum;
    }
}

// The forward function that sets up the grid and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Grid: each block computes a tile of size (TILE_M x TILE_N) output elements
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    // Block: each warp is 32 threads, and we need TILE_M*TILE_N warps per block
    dim3 block(32, TILE_M * TILE_N);

    // Calculate shared memory size: for each warp in the block, we allocate 2*CHUNK_SIZE floats
    size_t shared_mem = (TILE_M * TILE_N * 2 * CHUNK_SIZE) * sizeof(float);

    matmul_transposed_kernel_optimized<<<grid, block, shared_mem>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matmul with transposed B using warp-level reduction (CUDA)");
}
