#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#define TILE_WIDTH 16
#define CONST_B_MAX_ELEMENTS 16384

// Declare constant memory for matrix B for float and double types
__constant__ float d_B_f[CONST_B_MAX_ELEMENTS];
__constant__ double d_B_d[CONST_B_MAX_ELEMENTS];

// CUDA kernel for matrix multiplication using constant memory for matrix B
// A: [M x K], B (in constant memory): [K x N], C: [M x N]
template <typename scalar_t>
__global__ void matmul_const_B_kernel(const scalar_t* __restrict__ A,
                                        scalar_t* __restrict__ C,
                                        int M, int K, int N) {
    __shared__ scalar_t tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t sum = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Select the appropriate constant memory pointer based on type
    const scalar_t* B_const = nullptr;
    if constexpr (std::is_same<scalar_t, float>::value) {
        B_const = (const scalar_t*)d_B_f;
    } else {
        B_const = (const scalar_t*)d_B_d;
    }

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_WIDTH + threadIdx.x;
        int b_row = t * TILE_WIDTH + threadIdx.y;

        // Load tile from A from global memory
        if (row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + a_col]);
        else
            tile_A[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);

        // Load tile from B from constant memory
        if (b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B_const[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function exposed to Python via Pybind11
// It copies matrix B to constant memory and launches the kernel
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Ensure matrix B fits into constant memory
    TORCH_CHECK(B.numel() <= CONST_B_MAX_ELEMENTS, "Matrix B is too large to fit in constant memory");

    auto C = torch::empty({M, N}, A.options());

    // Copy matrix B to constant memory based on its type
    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "copy_B_to_constant", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(d_B_f, B.data_ptr<scalar_t>(), B.numel() * sizeof(scalar_t));
        } else {
            cudaMemcpyToSymbol(d_B_d, B.data_ptr<scalar_t>(), B.numel() * sizeof(scalar_t));
        }
    }));

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_const_B_kernel", ([&] {
        matmul_const_B_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication using constant memory for matrix B (CUDA)");
}
