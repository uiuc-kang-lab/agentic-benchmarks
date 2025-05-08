#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t fast_exp(scalar_t x) {
    return expf(x);
}

template <>
__device__ inline double fast_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__global__ void selu_shared_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  size_t numel) {
    constexpr scalar_t alpha = 1.67326324235437728481;
    constexpr scalar_t lambda = 1.05070098735548049342;
    constexpr int tile_size = 128;
    
    __shared__ scalar_t smem[tile_size];
    
    int tid = threadIdx.x;
    for (size_t pos = blockIdx.x * tile_size; pos < numel; pos += gridDim.x * tile_size) {
        int index = pos + tid;
        if (index < numel) {
            smem[tid] = input[index];
        }
        __syncthreads();

        if (index < numel) {
            scalar_t x = smem[tid];
            scalar_t act = (x > scalar_t(0)) ? x : alpha * (fast_exp(x) - scalar_t(1));
            output[index] = lambda * act;
        }
        __syncthreads();
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA tensor");
    
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    constexpr int threads = 128;
    int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_shared", ([&] {
        selu_shared_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Forward (Shared Memory Optimized)");
}