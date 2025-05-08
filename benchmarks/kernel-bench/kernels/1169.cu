#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 32

// Align memory accesses to 128-bit boundaries
#define ALIGNMENT 16

// Use __ldg for read-only accesses
template <typename T>
__device__ inline T load(const T* ptr) {
    return __ldg(ptr);
}

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

        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            scalar_t a_val = load(&A[global_n * M * K + m * K + k]);
            scalar_t b_val = load(&B[k * L + l]);
            sum += a_val * b_val;
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

    const int threads = 1024;
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