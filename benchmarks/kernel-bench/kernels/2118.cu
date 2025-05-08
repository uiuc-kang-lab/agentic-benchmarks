#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

__global__ void pipelined_triangular_mm_kernel(
    const float* __restrict__ A_chunk,
    const float* __restrict__ B_chunk,
    float* __restrict__ C_chunk,
    const int N,
    const int chunk_start
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tid_x = static_cast<int>(threadIdx.x);
    const int tid_y = static_cast<int>(threadIdx.y);
    const int global_row = static_cast<int>(blockIdx.y) * TILE_SIZE + tid_y + chunk_start;
    const int global_col = static_cast<int>(blockIdx.x) * TILE_SIZE + tid_x;

    float sum = 0.0f;

    for (int t = static_cast<int>(blockIdx.x); t <= static_cast<int>(blockIdx.y); ++t) {
        // Load tile from A
        int tRow = static_cast<int>(blockIdx.y) * TILE_SIZE + tid_y;
        int tCol = t * TILE_SIZE + tid_x;
        if ((tRow + chunk_start) < N && tCol < N && (tRow + chunk_start) >= tCol)
            As[tid_y][tid_x] = __ldg(&A_chunk[tRow * N + tCol]);
        else
            As[tid_y][tid_x] = 0.0f;

        // Load tile from B
        tRow = t * TILE_SIZE + tid_y;
        tCol = static_cast<int>(blockIdx.x) * TILE_SIZE + tid_x;
        if (tRow < N && tCol < N && tRow >= tCol)
            Bs[tid_y][tid_x] = __ldg(&B_chunk[tRow * N + tCol]);
        else
            Bs[tid_y][tid_x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[tid_y][k] * Bs[k][tid_x];
        }
        __syncthreads();
    }

    if (global_row < N && global_col < N) {
        if (global_row >= global_col) {
            C_chunk[global_row * N + global_col] = sum;
        } else {
            C_chunk[global_row * N + global_col] = 0.0f;
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 block(TILE_SIZE, TILE_SIZE);

    for (int chunk = 0; chunk < N; chunk += chunk_size) {
        const int current_chunk_size = std::min(chunk_size, N - chunk);
        const int stream_idx = (chunk / chunk_size) % NUM_STREAMS;
        
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);

        pipelined_triangular_mm_kernel<<<grid, block, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined triangular matrix multiplication (CUDA)");
}