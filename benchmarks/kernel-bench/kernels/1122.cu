#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    // Shared memory for B tile
    extern __shared__ char shared_mem[];
    scalar_t* b_shared = reinterpret_cast<scalar_t*>(shared_mem);
    
    // 2D block for M and L dimensions
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y * blockDim.y + threadIdx.y;

    // Process 2 elements per thread in M dimension when possible
    const int m2 = m + blockDim.x;
    
    // Process multiple N in grid-stride loop
    for (int n = 0; n < N; n++) {
        scalar_t sum1 = 0;
        scalar_t sum2 = m2 < M ? 0 : 0;

        // Process K in tiles
        for (int kt = 0; kt < K; kt += block_size) {
            // Collaborative loading of B tile into shared memory
            if (kt + tid < K && l < L) {
                b_shared[tid] = B[(kt + tid) * L + l];
            }
            __syncthreads();

            // Compute partial sums for this tile
            #pragma unroll 8
            for (int k = 0; k < min(block_size, K - kt); ++k) {
                if (m < M && l < L) {
                    scalar_t a_val1 = __ldg(&A[n * M * K + m * K + (kt + k)]);
                    sum1 += a_val1 * b_shared[k];
                    
                    if (m2 < M) {
                        scalar_t a_val2 = __ldg(&A[n * M * K + m2 * K + (kt + k)]);
                        sum2 += a_val2 * b_shared[k];
                    }
                }
            }
            __syncthreads();
        }

        // Store results
        if (m < M && l < L) {
            output[n * M * L + m * L + l] = sum1;
            if (m2 < M) {
                output[n * M * L + m2 * L + l] = sum2;
            }
        }
    }}
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // 2D thread block configuration
    dim3 threads(16, 16);
    dim3 blocks(
        (M + threads.x - 1) / threads.x,
        (L + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
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