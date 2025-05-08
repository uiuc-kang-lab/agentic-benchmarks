#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE = 32;
const int COARSE = 4;

__global__ void tile_matmul_transposed(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE * COARSE + tx;
    
    float4 cv[COARSE] = {make_float4(0, 0, 0, 0)};

    for (int t = 0; t < K; t += TILE) {
        // Vectorized tile loading
        int load_idx = t + tx;
        if (row < M && load_idx < K) {
            *((float4*)(&As[ty][tx*4])) = *((float4*)(A + row*K + load_idx*4));
        } else {
            *((float4*)(&As[ty][tx*4])) = make_float4(0, 0, 0, 0);
        }

        #pragma unroll
        for (int c = 0; c < COARSE; ++c) {
            int b_col = col + c*TILE;
            if (b_col < N && load_idx < K) {
                *((float4*)(&Bs[ty][tx*4])) = *((float4*)(B + b_col*K + load_idx*4));
            } else {
                *((float4*)(&Bs[ty][tx*4])) = make_float4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float4 a = *((float4*)(&As[ty][k*4]));
            #pragma unroll
            for (int c = 0; c < COARSE; ++c) {
                float4 b = *((float4*)(&Bs[k][c*TILE*4 + tx*4]));
                cv[c].x += a.x * b.x;
                cv[c].y += a.y * b.y;
                cv[c].z += a.z * b.z;
                cv[c].w += a.w * b.w;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < COARSE; ++c) {
        int write_col = col + c*TILE;
        if (row < M && write_col < N) {
            atomicAdd(&C[row*N + write_col], cv[c].x + cv[c].y + cv[c].z + cv[c].w);
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Dimension mismatch");

    int M = A.size(0);
    int N = B.size(0);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 grid((N + TILE*COARSE - 1) / (TILE*COARSE), (M + TILE - 1) / TILE);
    dim3 block(TILE/4, TILE);

    tile_matmul_transposed<<<grid, block>>>(A.data_ptr<float>(), 
                                           B.data_ptr<float>(), 
                                           C.data_ptr<float>(),
                                           M, N, K);
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled matrix multiply with vector coarsening");
}