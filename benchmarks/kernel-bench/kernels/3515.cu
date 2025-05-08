#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t my_exp(scalar_t x);

template <>
__device__ __forceinline__ float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ __forceinline__ double my_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__global__ void selu_kernel_shared(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const size_t numel) {
    __shared__ scalar_t shared_data[256];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    const scalar_t alpha = 1.67326324235437728481;
    const scalar_t lambda = 1.05070098735548049342;
    
    for (int idx = gid; idx < numel; idx += stride) {
        // Load data into shared memory
        shared_data[tid] = __ldg(&input[idx]);
        __syncthreads();
        
        // Process data from shared memory
        const scalar_t x = shared_data[tid];
        const scalar_t result = (x > static_cast<scalar_t>(0))
            ? x
            : alpha * (my_exp(x) - static_cast<scalar_t>(1));
        output[idx] = lambda * result;
        
        __syncthreads();
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_shared<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with Shared Memory (CUDA)");
}