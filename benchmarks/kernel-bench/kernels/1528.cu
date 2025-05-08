#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define UNROLL_FACTOR 4

typedef float4 vec4;

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    __shared__ vec4 s_A[2][TILE_SIZE][TILE_SIZE/4];
    __shared__ vec4 s_B[2][TILE_SIZE][TILE_SIZE/4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int buffer = 0;
    
    float4 accum[UNROLL_FACTOR] = {make_float4(0, 0, 0, 0)};
    
    for (int row = blockIdx.y * TILE_SIZE; row < N; row += TILE_SIZE * gridDim.y) {
        for (int col = blockIdx.x * TILE_SIZE; col < N; col += TILE_SIZE * gridDim.x) {
            float sum = 0.0f;
            
            // Prefetch first tile
            if (row + ty < N && col + tx*4 < N) {
                const vec4* a_ptr = reinterpret_cast<const vec4*>(A + (row + ty)*N + col);
                s_A[buffer][ty][tx] = (col + tx*4 + 3 < N) ? a_ptr[tx] : vec4{0,0,0,0};
                
                const vec4* b_ptr = reinterpret_cast<const vec4*>(B + (row + ty)*N + col);
                s_B[buffer][ty][tx] = (col + tx*4 + 3 < N) ? b_ptr[tx] : vec4{0,0,0,0};
            }
            __syncthreads();

            for (int i = 0; i < (N + TILE_SIZE - 1)/TILE_SIZE; ++i) {
                // Prefetch next tile
                if (i < (N + TILE_SIZE - 1)/TILE_SIZE - 1) {
                    int next_col = (i + 1) * TILE_SIZE + tx*4;
                    if (row + ty < N && next_col < N) {
                        const vec4* a_ptr = reinterpret_cast<const vec4*>(A + (row + ty)*N + next_col);
                        s_A[1 - buffer][ty][tx] = (next_col + 3 < N) ? a_ptr[0] : vec4{0,0,0,0};
                        
                        const vec4* b_ptr = reinterpret_cast<const vec4*>(B + next_col*N + col + ty*4);
                        s_B[1 - buffer][ty][tx] = (next_col + 3 < N) ? b_ptr[0] : vec4{0,0,0,0};
                    }
                }

                // Compute current tile
                #pragma unroll
                for (int k = 0; k < TILE_SIZE/4; k++) {
                    vec4 a = s_A[buffer][ty][k];
                    vec4 b = s_B[buffer][k][tx];
                    
                    sum += a.x * b.x;
                    sum += a.y * b.y;
                    sum += a.z * b.z;
                    sum += a.w * b.w;
                }
                
                __syncthreads();
                buffer = 1 - buffer;
            }

            if (row + ty < N && col + tx*4 < N) {
                float* out = C + (row + ty)*N + col + tx*4;
                if (col + tx*4 + 3 < N) {
                    reinterpret_cast<vec4*>(out)[0] = vec4{sum, sum, sum, sum};
                } else {
                    for (int k = 0; k < 4 && (col + tx*4 + k) < N; k++)
                        out[k] = sum;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE_SIZE/4, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Double Buffering");
}