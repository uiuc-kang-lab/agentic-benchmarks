#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel computes the product C = tril(A * B) for lower triangular matrices A and B.
// Each warp cooperatively computes one output element C(r, c) = sum_{k=c}^{r} A[r, k] * B[k, c] using a tiled load
// of the B-vector into shared memory and warp-level reduction via __shfl_down_sync().

// The lower triangular elements are linearly indexed as L = r*(r+1)/2 + c, with 0 <= c <= r < N.
// We invert this mapping using the quadratic formula.

__global__ void triangular_mm_warp_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    // Global thread id and warp id (each warp has 32 threads)
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / 32;
    int lane = globalThreadId % 32;

    // Total number of lower triangular elements
    int total = N * (N + 1) / 2;
    if (warpId >= total) return;

    // Map warpId to matrix indices (r, c) for lower triangular part.
    // Solve: warpId = r*(r+1)/2 + c, 0 <= c <= r. Invert using quadratic formula.
    float warpId_f = static_cast<float>(warpId);
    int r = int(floor((sqrtf(8.f * warpId_f + 1.f) - 1.f) / 2.f));
    int base = r * (r + 1) / 2;
    int c = warpId - base;

    // Each output element is computed as: C[r, c] = sum_{k=c}^{r} A[r*N + k] * B[k*N + c]
    float sum = 0.f;
    int len = r - c + 1;  // number of summation terms

    // We'll process the summation in chunks of TILE_SZ elements
    const int TILE_SZ = 32;

    // Declare shared memory for tiles of A and B values
    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = &shared[TILE_SZ];
    int warp_shared_offset = (threadIdx.x / 32) * TILE_SZ;

    // Loop in chunks over the reduction index
    for (int chunk = 0; chunk < len; chunk += TILE_SZ) {
        int k = c + chunk + lane;  // each lane processes one element in this chunk
        
        // Coalesced loads from global memory
        float a_val = 0.f, b_val = 0.f;
        if (k < c + len && k <= r && k < N) {
            a_val = A[r * N + k];
            b_val = B[k * N + c];
        }
        
        // Load values into shared memory
        sA[warp_shared_offset + lane] = a_val;
        sB[warp_shared_offset + lane] = b_val;
        __syncwarp();  // Only synchronize within the warp

        // Each thread multiplies its values if within bounds
        if (k < c + len && k <= r && k < N) {
            sum += sA[warp_shared_offset + lane] * sB[warp_shared_offset + lane];
        }
        __syncwarp();  // Ensure shared memory is ready for next iteration
    }

    // Now perform a warp-level reduction so that lane 0 accumulates the full sum from all lanes in the warp.
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Lane 0 writes the computed dot product to the output matrix
    if (lane == 0) {
        C[r * N + c] = sum;
    }
}

// C++ interface exposed to PyTorch via pybind11
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Total number of lower triangular elements
    int total = N * (N + 1) / 2;
    // Each warp (32 threads) computes one output element, so total threads = total * 32
    int threads_per_block = 128; // e.g., 4 warps per block
    int num_blocks = (total * 32 + threads_per_block - 1) / threads_per_block;

    // Shared memory: each block has (threads_per_block / 32) warps, each needs TILE_SZ floats
    int shared_mem_size = (threads_per_block / 32) * 32 * sizeof(float);

    triangular_mm_warp_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with warp-level reduction (CUDA)");
}
