#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#define TILE_WIDTH 32

// Vectorized load traits: for float use float4 (4 floats), for double use double2 (2 doubles)
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    using VecT = float4;
    static constexpr int vec_size = 4;
};

template <>
struct VecTraits<double> {
    using VecT = double2;
    static constexpr int vec_size = 2;
};

// CUDA kernel for matrix multiplication with vectorized global memory loads using __ldg()
// This kernel assumes the input matrices are 128-bit aligned and uses a tile size of 32

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C, int M, int K, int N) {
    // Shared memory for A and B tiles
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // Row index for C
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // Column index for C

    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    constexpr int vec_size = VecTraits<scalar_t>::vec_size;

    for (int t = 0; t < numTiles; t++) {
        // Load tile from matrix A into shared memory
        int A_tile_col_base = t * TILE_WIDTH;
        int global_A_col = A_tile_col_base + threadIdx.x;

        // Only one thread per group of vec_size in x dimension does the vectorized load
        if ((threadIdx.x % vec_size) == 0) {
            if (row < M) {
                const scalar_t* A_row = A + row * K;
                const typename VecTraits<scalar_t>::VecT* A_vec_ptr = reinterpret_cast<const typename VecTraits<scalar_t>::VecT*>(A_row);
                int vec_index = (A_tile_col_base + threadIdx.x) / vec_size;
                typename VecTraits<scalar_t>::VecT vecData;
                if (A_tile_col_base + threadIdx.x + vec_size - 1 < K) {
                    vecData = __ldg(&A_vec_ptr[vec_index]);
                } else {
                    scalar_t temp[vec_size];
                    for (int i = 0; i < vec_size; i++) {
                        int idx = A_tile_col_base + threadIdx.x + i;
                        temp[i] = (idx < K) ? __ldg(&A_row[idx]) : scalar_t(0);
                    }
                    if constexpr (std::is_same<scalar_t, float>::value) {
                        vecData = make_float4(temp[0], temp[1], temp[2], temp[3]);
                    } else if constexpr (std::is_same<scalar_t, double>::value) {
                        vecData = make_double2(temp[0], temp[1]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < vec_size; i++) {
                    int shared_col = threadIdx.x + i;
                    if (shared_col < TILE_WIDTH) {
                        if constexpr (std::is_same<scalar_t, float>::value) {
                            float4 v = vecData;
                            float elem = (i==0) ? v.x : (i==1) ? v.y : (i==2) ? v.z : v.w;
                            sA[threadIdx.y][shared_col] = elem;
                        } else if constexpr (std::is_same<scalar_t, double>::value) {
                            double2 v = vecData;
                            double elem = (i==0) ? v.x : v.y;
                            sA[threadIdx.y][shared_col] = elem;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < vec_size; i++) {
                    int shared_col = threadIdx.x + i;
                    if (shared_col < TILE_WIDTH) {
                        sA[threadIdx.y][shared_col] = scalar_t(0);
                    }
                }
            }
        }
        // Load tile from matrix B into shared memory
        int B_tile_row = t * TILE_WIDTH + threadIdx.y; // Row index in B
        int B_tile_col_base = blockIdx.x * TILE_WIDTH;
        int global_B_col = B_tile_col_base + threadIdx.x;
        if ((threadIdx.x % vec_size) == 0) {
            if (B_tile_row < K) {
                const scalar_t* B_row = B + B_tile_row * N;
                const typename VecTraits<scalar_t>::VecT* B_vec_ptr = reinterpret_cast<const typename VecTraits<scalar_t>::VecT*>(B_row);
                int vec_index = (B_tile_col_base + threadIdx.x) / vec_size;
                typename VecTraits<scalar_t>::VecT vecData;
                if (B_tile_col_base + threadIdx.x + vec_size - 1 < N) {
                    vecData = __ldg(&B_vec_ptr[vec_index]);
                } else {
                    scalar_t temp[vec_size];
                    for (int i = 0; i < vec_size; i++) {
                        int idx = B_tile_col_base + threadIdx.x + i;
                        temp[i] = (idx < N) ? __ldg(&B_row[idx]) : scalar_t(0);
                    }
                    if constexpr (std::is_same<scalar_t, float>::value) {
                        vecData = make_float4(temp[0], temp[1], temp[2], temp[3]);
                    } else if constexpr (std::is_same<scalar_t, double>::value) {
                        vecData = make_double2(temp[0], temp[1]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < vec_size; i++) {
                    int shared_col = threadIdx.x + i;
                    if (shared_col < TILE_WIDTH) {
                        if constexpr (std::is_same<scalar_t, float>::value) {
                            float4 v = vecData;
                            float elem = (i==0) ? v.x : (i==1) ? v.y : (i==2) ? v.z : v.w;
                            sB[threadIdx.y][shared_col] = elem;
                        } else if constexpr (std::is_same<scalar_t, double>::value) {
                            double2 v = vecData;
                            double elem = (i==0) ? v.x : v.y;
                            sB[threadIdx.y][shared_col] = elem;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < vec_size; i++) {
                    int shared_col = threadIdx.x + i;
                    if (shared_col < TILE_WIDTH) {
                        sB[threadIdx.y][shared_col] = scalar_t(0);
                    }
                }
            }
        }

        __syncthreads();

        // Compute dot product for the current tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Pybind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Vectorized optimized matrix multiplication forward (CUDA, __ldg, 128-bit aligned)");
}
