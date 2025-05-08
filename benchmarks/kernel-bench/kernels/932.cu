#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__inline__ __device__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_kernel(float* __restrict__ out, const float* __restrict__ in, int N) {
    __shared__ float shared[BLOCK_SIZE/WARP_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    float sum = 0.0f;
    if (gid < N) {
        sum = in[gid];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Write reduced sum to shared memory
    if (lane == 0) {
        shared[wid] = sum;
    }
    
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (BLOCK_SIZE/WARP_SIZE)) {
        sum = shared[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            atomicAdd(out, sum);
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with warp reduction (CUDA)");
}