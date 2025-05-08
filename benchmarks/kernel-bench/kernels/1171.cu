#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 32
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L,
    const int chunk_start) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * WARP_SIZE + tid;
    const int chunk_elements = CHUNK_SIZE * M * L;
    
    if (idx < chunk_elements) {
        const int local_n = idx / (M * L);
        const int m = (idx % (M * L)) / L;
        const int l = idx % L;
        
        const int global_n = chunk_start + local_n;
        if (global_n >= N) return;

        const int a_base = global_n * M * K + m * K;
        
        scalar_t sum = 0;
        
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            scalar_t a_val = __ldg(&A[a_base + k]);
            scalar_t b_val = __ldg(&B[k * L + l]);  // Fixed index calculation for B
            sum = fma(a_val, b_val, sum);  // Use fma for better numerical stability
        }
        
        output[global_n * M * L + m * L + l] = sum;
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

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    const int elements_per_chunk = CHUNK_SIZE * M * L;
    const int blocks = (elements_per_chunk + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
            const int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
            
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
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

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