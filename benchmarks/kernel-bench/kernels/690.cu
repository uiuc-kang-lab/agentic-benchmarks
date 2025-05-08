#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float4 vec4;

template <typename scalar_t>
__global__ void vectorized_matmul_kernel(const scalar_t* __restrict__ A,
                                        const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C,
                                        int M, int K, int N) {
    const int TILE = 32;
    const int TILE_PER_THREAD = 2;
    __shared__ scalar_t sA[TILE][TILE];
    __shared__ scalar_t sB[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty * TILE_PER_THREAD;
    int col = bx * TILE + tx * TILE_PER_THREAD;

    scalar_t c[TILE_PER_THREAD][TILE_PER_THREAD] = {{0}};

    for (int t = 0; t < K; t += TILE) {
        // Vectorized loading of A tile
        if (row + TILE_PER_THREAD <= M && t + tx * TILE_PER_THREAD + TILE <= K) {
            vec4 a = reinterpret_cast<const vec4*>(&A[row * K + t + tx * TILE_PER_THREAD])[0];
            sA[ty * TILE_PER_THREAD][tx * TILE_PER_THREAD] = a.x;
            sA[ty * TILE_PER_THREAD][tx * TILE_PER_THREAD + 1] = a.y;
            sA[ty * TILE_PER_THREAD + 1][tx * TILE_PER_THREAD] = a.z;
            sA[ty * TILE_PER_THREAD + 1][tx * TILE_PER_THREAD + 1] = a.w;
        } else {
            for (int i = 0; i < TILE_PER_THREAD; ++i)
                for (int j = 0; j < TILE_PER_THREAD; ++j)
                    sA[ty * TILE_PER_THREAD + i][tx * TILE_PER_THREAD + j] = 
                        (row + i < M && t + tx * TILE_PER_THREAD + j < K) 
                        ? A[(row + i) * K + t + tx * TILE_PER_THREAD + j] : 0;
        }

        // Vectorized loading of B tile
        if (t + ty * TILE_PER_THREAD + TILE <= K && col + TILE_PER_THREAD <= N) {
            vec4 b = reinterpret_cast<const vec4*>(&B[(t + ty * TILE_PER_THREAD) * N + col])[0];
            sB[ty * TILE_PER_THREAD][tx * TILE_PER_THREAD] = b.x;
            sB[ty * TILE_PER_THREAD][tx * TILE_PER_THREAD + 1] = b.y;
            sB[ty * TILE_PER_THREAD + 1][tx * TILE_PER_THREAD] = b.z;
            sB[ty * TILE_PER_THREAD + 1][tx * TILE_PER_THREAD + 1] = b.w;
        } else {
            for (int i = 0; i < TILE_PER_THREAD; ++i)
                for (int j = 0; j < TILE_PER_THREAD; ++j)
                    sB[ty * TILE_PER_THREAD + i][tx * TILE_PER_THREAD + j] = 
                        (t + ty * TILE_PER_THREAD + i < K && col + j < N)
                        ? B[(t + ty * TILE_PER_THREAD + i) * N + col + j] : 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            scalar_t a0 = sA[ty * TILE_PER_THREAD][k];
            scalar_t a1 = sA[ty * TILE_PER_THREAD + 1][k];
            scalar_t b0 = sB[k][tx * TILE_PER_THREAD];
            scalar_t b1 = sB[k][tx * TILE_PER_THREAD + 1];

            c[0][0] += a0 * b0;
            c[0][1] += a0 * b1;
            c[1][0] += a1 * b0;
            c[1][1] += a1 * b1;
        }
        __syncthreads();
    }

    if (row + TILE_PER_THREAD <= M && col + TILE_PER_THREAD <= N) {
        for (int i = 0; i < TILE_PER_THREAD; ++i)
            for (int j = 0; j < TILE_PER_THREAD; ++j)
                C[(row + i) * N + col + j] = c[i][j];
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Invalid dimensions");

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 31) / 32, (M + 31) / 32);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "vectorized_matmul", [&] {
        vectorized_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    });

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Vectorized tile matrix multiplication");
}