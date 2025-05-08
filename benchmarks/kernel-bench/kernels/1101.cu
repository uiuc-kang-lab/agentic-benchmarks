#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void highly_unrolled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int batch = row / M;
    const int m = row % M;

    __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    
    // Main computation loop, unrolled by UNROLL_FACTOR
    #pragma unroll 1
    for (int t = 0; t < K; t += TILE_DIM) {
        // Load tiles into shared memory with manual unrolling
        if (batch < N && m < M) {
            #pragma unroll
            for (int u = 0; u < TILE_DIM; u += UNROLL_FACTOR) {
                if ((t + u) < K) {
                    scalar_t* sA_ptr = &sA[threadIdx.y][u];
                    const scalar_t* A_ptr = &A[batch * M * K + m * K + t + u];
                    
                    #pragma unroll
                    for (int v = 0; v < UNROLL_FACTOR; v++) {
                        if ((t + u + v) < K) {
                            sA_ptr[v] = __ldg(A_ptr + v);
                        }
                    }
                }
            }
        }

        if (col < L) {
            #pragma unroll
            for (int u = 0; u < TILE_DIM; u += UNROLL_FACTOR) {
                if ((t + u) < K) {
                    scalar_t* sB_ptr = &sB[u][threadIdx.x];
                    const scalar_t* B_ptr = &B[(t + u) * L + col];
                    
                    #pragma unroll
                    for (int v = 0; v < UNROLL_FACTOR; v++) {
                        if ((t + u + v) < K) {
                            sB_ptr[v * TILE_DIM] = __ldg(B_ptr + v * L);
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Compute partial results with aggressive unrolling
        if ((t + TILE_DIM) <= K) {
            // Full tile computation
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i += UNROLL_FACTOR) {
                scalar_t a0 = sA[threadIdx.y][i];
                scalar_t a1 = sA[threadIdx.y][i+1];
                scalar_t a2 = sA[threadIdx.y][i+2];
                scalar_t a3 = sA[threadIdx.y][i+3];

                scalar_t b0 = sB[i][threadIdx.x];
                scalar_t b1 = sB[i+1][threadIdx.x];
                scalar_t b2 = sB[i+2][threadIdx.x];
                scalar_t b3 = sB[i+3][threadIdx.x];

                sum = __fmaf_rn(a0, b0, sum);
                sum = __fmaf_rn(a1, b1, sum);
                sum = __fmaf_rn(a2, b2, sum);
                sum = __fmaf_rn(a3, b3, sum);
            }
        } else {
            // Boundary case
            #pragma unroll
            for (int i = 0; i < TILE_DIM && (t + i) < K; i++) {
                sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write output with bounds checking
    if (batch < N && m < M && col < L) {
        output[batch * M * L + m * L + col] = sum;
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

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "highly_unrolled_kernel", ([&] {
        highly_unrolled_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Highly unrolled tensor-matrix multiplication (CUDA)");
}