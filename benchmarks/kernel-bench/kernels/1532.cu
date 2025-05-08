#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define UNROLL_FACTOR 4

typedef float4 vec4;

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    __shared__ vec4 s_A[2][TILE_SIZE][TILE_SIZE/4];
    __shared__ vec4 s_B[2][TILE_SIZE][TILE_SIZE/4];

    // Each block computes a TILE_SIZE x TILE_SIZE submatrix of C
    int tx = threadIdx.x; // Each thread handles 4 columns
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx * 4;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int buffer = 0;
    float sum = 0.0f;

    // Prefetch the first tile (tile 0) from A and B
    int a_col = 0 * TILE_SIZE + tx * 4;
    if (row < N && a_col < N) {
        const vec4* a_ptr = reinterpret_cast<const vec4*>(A + row * N + a_col);
        s_A[buffer][ty][tx] = (a_col + 3 < N) ? a_ptr[0] : vec4{0, 0, 0, 0};
    } else {
        s_A[buffer][ty][tx] = vec4{0, 0, 0, 0};
    }

    int b_row = 0 * TILE_SIZE + ty;
    if (b_row < N && col < N) {
        const vec4* b_ptr = reinterpret_cast<const vec4*>(B + b_row * N + col);
        s_B[buffer][ty][tx] = (col + 3 < N) ? b_ptr[0] : vec4{0, 0, 0, 0};
    } else {
        s_B[buffer][ty][tx] = vec4{0, 0, 0, 0};
    }
    __syncthreads();

    // Loop over the tiles of the input matrices
    for (int tile = 0; tile < numTiles; tile++) {
        int next_buffer = 1 - buffer;
        if (tile < numTiles - 1) {
            int next_tile = tile + 1;
            int next_a_col = next_tile * TILE_SIZE + tx * 4;
            if (row < N && next_a_col < N) {
                const vec4* a_ptr = reinterpret_cast<const vec4*>(A + row * N + next_a_col);
                s_A[next_buffer][ty][tx] = (next_a_col + 3 < N) ? a_ptr[0] : vec4{0, 0, 0, 0};
            } else {
                s_A[next_buffer][ty][tx] = vec4{0, 0, 0, 0};
            }
            int next_b_row = next_tile * TILE_SIZE + ty;
            if (next_b_row < N && col < N) {
                const vec4* b_ptr = reinterpret_cast<const vec4*>(B + next_b_row * N + col);
                s_B[next_buffer][ty][tx] = (col + 3 < N) ? b_ptr[0] : vec4{0, 0, 0, 0};
            } else {
                s_B[next_buffer][ty][tx] = vec4{0, 0, 0, 0};
            }
        }
        __syncthreads();

        // Compute the product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE/4; k++) {
            vec4 a = s_A[buffer][ty][k];
            vec4 b = s_B[buffer][k][tx];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        __syncthreads();
        buffer = 1 - buffer; // Swap buffers
    }

    // Write the result to C
    if (row < N && col < N) {
        float* out = C + row * N + col;
        if (col + 3 < N) {
            reinterpret_cast<vec4*>(out)[0] = vec4{sum, sum, sum, sum};
        } else {
            for (int k = 0; k < 4 && (col + k) < N; k++)
                out[k] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE_SIZE/4, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Double Buffering");
}