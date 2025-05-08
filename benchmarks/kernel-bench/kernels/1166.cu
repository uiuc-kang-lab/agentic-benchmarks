#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 32
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L,
    int chunk_start) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_elements = CHUNK_SIZE * M * L;
    
    if (idx < chunk_elements) {
        int local_n = idx / (M * L);
        int m = (idx % (M * L)) / L;
        int l = idx % L;
        
        int global_n = chunk_start + local_n;
        if (global_n >= N) return;

        // Pre-compute base indices to reduce address calculations
        const int base_a = global_n * M * K + m * K;
        const int base_b = l;
        
        scalar_t sum = 0;
        
        // Manual unrolling of K-dimension loop
        int k = 0;
        #pragma unroll
        for (; k + UNROLL_FACTOR <= K; k += UNROLL_FACTOR) {
            scalar_t a0 = A[base_a + k];
            scalar_t a1 = A[base_a + k + 1];
            scalar_t a2 = A[base_a + k + 2];
            scalar_t a3 = A[base_a + k + 3];
            
            scalar_t b0 = B[k * L + base_b];
            scalar_t b1 = B[(k + 1) * L + base_b];
            scalar_t b2 = B[(k + 2) * L + base_b];
            scalar_t b3 = B[(k + 3) * L + base_b];
            
            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }
        
        // Handle remaining iterations
        #pragma unroll
        for (; k < K; k++) {
            sum += A[base_a + k] * B[k * L + base_b];
        }
        
        output[global_n * M * L + m * L + l] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    const int elements_per_chunk = CHUNK_SIZE * M * L;
    const int blocks = (elements_per_chunk + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
            int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
            
            module_fn_cuda_kernel<scalar_t><<<blocks, threads, 0, streams[stream_idx]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M, K, L,
                chunk_start);
        }
    }));

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}