#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void pipelined_stream_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L,
    int chunk_offset, int chunk_size) {

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Adjust row based on chunk offset
    row += chunk_offset;

    __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM];
    
    scalar_t sum = 0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        if (row < chunk_offset + chunk_size && row < (N * M) && (t * TILE_DIM + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + t * TILE_DIM + threadIdx.x]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < L) {
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[(t * TILE_DIM + threadIdx.y) * L + col]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < chunk_offset + chunk_size && row < (N * M) && col < L) {
        output[row * L + col] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for each stream
    const int total_rows = N * M;
    const int chunk_size = (total_rows + NUM_STREAMS - 1) / NUM_STREAMS;
    
    dim3 threads(TILE_DIM, TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "pipelined_stream_kernel", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int chunk_offset = i * chunk_size;
            int current_chunk_size = min(chunk_size, total_rows - chunk_offset);
            
            if (current_chunk_size <= 0) break;

            // Calculate grid dimensions for this chunk
            dim3 grid((L + TILE_DIM - 1) / TILE_DIM, 
                     (current_chunk_size + TILE_DIM - 1) / TILE_DIM);

            pipelined_stream_kernel<scalar_t><<<grid, threads, 0, streams[i]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M, K, L,
                chunk_offset, current_chunk_size
            );
        }
    }));

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int N = A.size(0);
    const int M = A.size(1);
    const int L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Pipelined stream tensor-matrix multiplication (CUDA)");
}