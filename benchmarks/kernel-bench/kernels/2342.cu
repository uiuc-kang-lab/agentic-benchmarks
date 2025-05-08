#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;
const int NUM_STREAMS = 4;
const int CHUNK_SIZE = 1024;

__global__ void matmul_transposed_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K, int m_offset) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int m = by * TILE_SIZE + ty + m_offset;
    int n = bx * TILE_SIZE + tx;

    float c_val = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        if (m < M && (k_offset + tx) < K) {
            As[ty][tx] = A[(m - m_offset) * K + k_offset + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (n < N && (k_offset + ty) < K) {
            Bs[ty][tx] = B[n * K + k_offset + ty];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = c_val;
    }
}

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
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(TILE_SIZE, TILE_SIZE);
    
    for (int m_offset = 0; m_offset < M; m_offset += CHUNK_SIZE) {
        int current_chunk_size = std::min(CHUNK_SIZE, M - m_offset);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);
                  (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);
        
        int stream_idx = (m_offset / CHUNK_SIZE) % NUM_STREAMS;
        
        matmul_transposed_kernel<<<grid, block, 0, streams[stream_idx]>>>(
            A.data_ptr<float>() + (m_offset * K),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K, m_offset
        );
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B (CUDA)");
}