#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

template <>
__device__ inline double gelu_function<double>(double x) {
    return x * 0.5 * (1.0 + erf(x * 0.7071067811865475));
}

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_shared_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  size_t numel) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(shared_mem);

    const int tid = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
    const int block_tid = threadIdx.x * VEC_SIZE;
    const int shared_size = blockDim.x * VEC_SIZE;

    // Coalesced global to shared memory load
    if (tid + VEC_SIZE <= numel) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            smem[block_tid + i] = input[tid + i];
        }
    } else {
        for (int i = 0; i < VEC_SIZE; ++i) {
            if (tid + i < numel) smem[block_tid + i] = input[tid + i];
        }
    }

    __syncthreads();

    // Process elements from shared memory
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (block_tid + i < shared_size && tid + i < numel) {
            smem[block_tid + i] = gelu_function(smem[block_tid + i]);
        }
    }

    __syncthreads();

    // Coalesced shared to global memory store
    if (tid + VEC_SIZE <= numel) {
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            output[tid + i] = smem[block_tid + i];
        }
    } else {
        for (int i = 0; i < VEC_SIZE; ++i) {
            if (tid + i < numel) output[tid + i] = smem[block_tid + i];
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t numel = x.numel();

    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int blocks = (numel + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    const size_t shared_mem = threads * VEC_SIZE * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_shared", [&] {
        gelu_shared_kernel<scalar_t, VEC_SIZE>
            <<<blocks, threads, shared_mem>>>(x.data_ptr<scalar_t>(),
                                            y.data_ptr<scalar_t>(),
                                            numel);
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with shared memory (CUDA)");
}