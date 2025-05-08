#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes the lower-triangular matrix product C = A * B where
// A and B are lower-triangular matrices. Each block is assigned one row of C.
// Within each block, each warp computes one output element C[r, c] in that row.
// The entire A-row (for row r) is loaded cooperatively into shared memory, so that
// every warp computing an element in that row can quickly access A[r,*].
// Then, each warp performs a reduction over the k dimension (from c to r) to compute:
//    C[r, c] = sum_{k=c}^{r} A[r,k] * B[k, c]
// The reduction is parallelized within the warp. Each thread in the warp takes a strided
// portion of the k-range, accumulates a partial sum, and the final reduction is done using
// warp-level shuffles (__shfl_down_sync).

// Kernel assumes blockDim.x = 32 (warpSize) and blockDim.y controls number of output elements
// computed per row. Grid.x is set to N (one block per row) and grid.y is chosen so that
// grid.y * blockDim.y >= N, covering all possible column indices (0 <= c < N). For any given
// row r, only indices c <= r hold valid lower-triangular outputs; for c > r the kernel writes 0.

__global__ void warp_reduce_triangular_mm_shared(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    // Each block is dedicated to one row r of the output
    int r = blockIdx.x;
    if (r >= N) return;

    // Each block computes a set of output columns for row r.
    // Compute output column index c using gridDim.y and blockDim.y
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    // Allocate shared memory for A[r,*]. Size is allocated dynamically as passed from the host.
    extern __shared__ float A_shared[];  // size: N floats per block

    // Load A[r, 0..r] into shared memory. We load the entire row up to N even though only [0, r] is used.
    // Use all threads in the x-dimension to load in a coalesced manner.
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        if (k <= r)
            A_shared[k] = A[r * N + k];
        else
            A_shared[k] = 0.0f;  // Fill unused entries with zero
    }
    __syncthreads();

    // Only process valid columns: for lower triangular, we require c <= r and c < N
    if (c >= N) return;  // Out-of-bound column
    if (c > r) {
        // For elements in the upper-triangular part, set to 0 (only one thread per element writes)
        if (threadIdx.x == 0) {
            C[r * N + c] = 0.0f;
        }
        return;
    }

    // Now, compute C[r, c] = sum_{k=c}^{r} A[r,k] * B[k, c]
    // We perform the reduction cooperatively within a warp.
    int lane = threadIdx.x;  // lane id in the warp; blockDim.x is assumed to be 32
    float partial = 0.0f;

    // Each thread processes a strided segment of the k-range from k = c to r
    // Note: warpSize is 32
    for (int k = c + lane; k <= r; k += 32) {
        partial += A_shared[k] * B[k * N + c];
    }

    // Perform warp-level reduction using __shfl_down_sync
    // All 32 lanes of the warp participate, even if some lanes had no work
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    // The first lane in the warp writes the computed value
    if (lane == 0) {
        C[r * N + c] = partial;
    }
}

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

    // Launch configuration:
    //   - One block per row (grid.x = N)
    //   - Each block computes a set of output columns for that row.
    //   - blockDim.x is fixed to warpSize (32) for warp-level reduction,
    //     and blockDim.y is chosen (e.g., 8) so that grid.y covers all columns (0..N-1).
    int block_y = 8;
    dim3 block(32, block_y);
    int grid_y = (N + block_y - 1) / block_y;
    dim3 grid(N, grid_y);

    // Allocate shared memory: one row of A, of length N floats per block.
    size_t shared_mem_size = N * sizeof(float);

    warp_reduce_triangular_mm_shared<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &forward, "Warp-reduce triangular matrix multiplication using shared memory (CUDA)");
}
