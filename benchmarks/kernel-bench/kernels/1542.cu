#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for matrix multiply
#define TILE_SIZE 32

// Kernel using asynchronous copies (cp.async) and double buffering
__global__ void matmul_kernel_async(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    // Number of tiles to cover the matrix
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices for C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Allocate shared memory for double buffered tiles
    // We allocate two buffers for A and two for B
    // Total shared memory = 4 * TILE_SIZE * TILE_SIZE * sizeof(float)
    extern __shared__ float shared_mem[];
    float* s_A = shared_mem;  // size: 2 * TILE_SIZE * TILE_SIZE
    float* s_B = s_A + 2 * TILE_SIZE * TILE_SIZE;  // size: 2 * TILE_SIZE * TILE_SIZE

    int curr_buf = 0;  // Current buffer index (0 or 1)
    float value = 0.0f;

    // ---------- Preload first tile (tile 0) asynchronously into current buffer ----------
    int tile = 0;
    int a_col = tile * TILE_SIZE + tx;
    int b_row = tile * TILE_SIZE + ty;

    // Pointers for destination in shared memory for current buffer
    float* dest_A = s_A + curr_buf * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
    float* dest_B = s_B + curr_buf * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;

    // Pointers for source in global memory
    const float* src_A = A + row * N + a_col;
    const float* src_B = B + (tile * TILE_SIZE + ty) * N + col;

    if (row < N && a_col < N) {
        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n" 
                      :
                      : "l"(dest_A), "l"(src_A), "n"(4));
    } else {
        *dest_A = 0.0f;
    }
    if (b_row < N && col < N) {
        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                      :
                      : "r"(dest_B), "l"(src_B), "n"(4));
    } else {
        *dest_B = 0.0f;
    }
    // Commit pending asynchronous copies
    asm volatile ("cp.async.commit_group;\n" ::: "memory");
    __syncthreads();

    // -------------------- Pipelined loop over tiles --------------------
    // For each tile, prefetch the next tile while computing the current one
    for (tile = 0; tile < num_tiles - 1; tile++) {
        int next_buf = 1 - curr_buf;
        int next_tile = tile + 1;

        // Set up pointers for next tile
        a_col = next_tile * TILE_SIZE + tx;
        b_row = next_tile * TILE_SIZE + ty;

        dest_A = s_A + next_buf * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
        dest_B = s_B + next_buf * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
        src_A = A + row * N + next_tile * TILE_SIZE + tx;
        src_B = B + (next_tile * TILE_SIZE + ty) * N + col;

        if (row < N && a_col < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "l"(dest_A), "l"(src_A), "n"(4));
        } else {
            *dest_A = 0.0f;
        }
        if (b_row < N && col < N) {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(dest_B), "l"(src_B), "n"(4));
        } else {
            *dest_B = 0.0f;
        }
        asm volatile ("cp.async.commit_group;\n" ::: "memory");

        // Compute on the current tile (stored in buffer curr_buf)
        __syncthreads();
        float* tile_A = s_A + curr_buf * TILE_SIZE * TILE_SIZE;
        float* tile_B = s_B + curr_buf * TILE_SIZE * TILE_SIZE;
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_val = tile_A[ty * TILE_SIZE + k];
            float b_val = tile_B[k * TILE_SIZE + tx];
            value = fmaf(a_val, b_val, value);
        }
        __syncthreads();  // ensure the next tile has been loaded
        curr_buf = next_buf;  // swap buffers
    }

    // -------------------- Compute on the final tile --------------------
    __syncthreads();
    float* tile_A = s_A + curr_buf * TILE_SIZE * TILE_SIZE;
    float* tile_B = s_B + curr_buf * TILE_SIZE * TILE_SIZE;
    for (int k = 0; k < TILE_SIZE; k++) {
        float a_val = tile_A[ty * TILE_SIZE + k];
        float b_val = tile_B[k * TILE_SIZE + tx];
        value = fmaf(a_val, b_val, value);
    }

    // Write the output
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Check that A and B are CUDA tensors, 2D, square, and of matching size
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Calculate shared memory size: 4 buffers of TILE_SIZE x TILE_SIZE floats
    size_t sharedMemSize = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);

    // Launch the asynchronous pipelined kernel
    matmul_kernel_async<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Asynchronous Copy and Pipelining (CUDA)");
}
