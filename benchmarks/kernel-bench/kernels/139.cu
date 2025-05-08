#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)
#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t handle = nullptr;

__global__ void warp_optimized_matmul_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int M, const int N, const int K) {
    const int row = blockIdx.y * TILE_SIZE + (threadIdx.x / WARP_SIZE) * (TILE_SIZE/WARPS_PER_BLOCK) + (threadIdx.x % (TILE_SIZE/WARPS_PER_BLOCK));
    const int col = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    float sum = 0.0f;
    
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ float shared_data[WARPS_PER_BLOCK][WARP_SIZE];
    
    for (int tile = 0; tile < K; tile += WARP_SIZE) {
        float a_reg = (row < M && (tile + lane) < K) ? A[row * K + tile + lane] : 0.0f;
        float b_reg = ((tile + lane) < K && col < N) ? B[(tile + lane) * N + col] : 0.0f;
        
        #pragma unroll
        for (int k = 0; k < WARP_SIZE; ++k) {
            float a_bc = __shfl_sync(0xffffffff, a_reg, k);
            sum += a_bc * b_reg;
            b_reg = __shfl_up_sync(0xffffffff, b_reg, 1);
        }
    }
    
    if (lane < WARP_SIZE) {
        shared_data[warp_id][lane] = sum;
    }
    __syncthreads();
    
    if (lane < WARPS_PER_BLOCK) {
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            warp_sum += shared_data[i][lane];
        }
        
        if (row < M && col < N) {
            C[row * N + col] = warp_sum;
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 block(BLOCK_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        warp_optimized_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                      .dtype(A.dtype())
                      .device(A.device())
                      .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized matrix multiplication (CUDA)");
}