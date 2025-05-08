#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    
    
    const int tid = threadIdx.x;
    const int n = blockIdx.z;
    const int m = blockIdx.y;
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (l < L) {
        // Initialize shared memory
        shared_data[tid] = 0;
        
        // Compute partial sums
        for (int k = 0; k < K; k++) {
            if (k < K && l < L) {
                scalar_t a_val = A[n * M * K + m * K + k];
                scalar_t b_val = B[k * L + l];
                shared_data[tid] += a_val * b_val;
            }
        }
        __syncthreads();
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            shared_data[tid] += __shfl_down_sync(0xffffffff, shared_data[tid], offset);
        }
        
        // First thread in warp writes result
        if (tid % warpSize == 0 && l < L) {
            output[n * M * L + m * L + l] = shared_data[tid];
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    const int threads = 256;
    const int shared_memory_size = threads * sizeof(float);
    const dim3 blocks((L + threads - 1) / threads, M, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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