#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_K 4
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    const int n = blockIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int l = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (m >= M || l >= L || n >= N) return;

    scalar_t sum = 0;
    
    // Tile K dimension using shared memory for B
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        __shared__ scalar_t sharedB[TILE_K][16];  // 16 corresponds to blockDim.z

        // Load a tile of B into shared memory (only one thread per column does the load)
        if (threadIdx.y == 0) {
            for (int i = 0; i < TILE_K; i++) {
                int global_k = k_base + i;
                int global_l = blockIdx.z * blockDim.z + threadIdx.z;
                sharedB[i][threadIdx.z] = (global_k < K && global_l < L) ? B[global_k * L + global_l] : static_cast<scalar_t>(0);
            }
        }
        __syncthreads();

        int k_end = k_base + TILE_K < K ? k_base + TILE_K : K;
        for (int k = k_base; k < k_end; k++) {
            // Each thread computes using its own m and l indices
            sum += A[n*M*K + m*K + k] * sharedB[k - k_base][threadIdx.z];
        }
        __syncthreads();
    }

    output[n*M*L + m*L + l] = sum;
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // 3D grid: 1 block per N, each block covers MxL
    dim3 threads(1, 16, 16);  // Total 256 threads
    dim3 blocks(
        N,
        (M + threads.y - 1) / threads.y,
        (L + threads.z - 1) / threads.z
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>( 
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L
        );
    }));

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
