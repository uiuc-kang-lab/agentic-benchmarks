#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    constexpr int VEC_SIZE = 4;
    extern __shared__ __align__(sizeof(scalar_t) * VEC_SIZE) scalar_t s_data[];
    
    const int tid = threadIdx.x;
    const int vec_idx = blockIdx.x * (blockDim.x * VEC_SIZE) + tid * VEC_SIZE;

    scalar_t local_data[VEC_SIZE];
    if (vec_idx + VEC_SIZE <= size) {
        for(int i = 0; i < VEC_SIZE; ++i) {
            local_data[i] = input[vec_idx + i];
        }
    } else {
        for(int i = 0; i < VEC_SIZE; ++i) {
            local_data[i] = (vec_idx + i < size) ? input[vec_idx + i] : 0;
        }
    }

    #pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i) {
        s_data[tid * VEC_SIZE + i] = local_data[i];
    }
    __syncthreads();

    #pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i) {
        const scalar_t x = s_data[tid * VEC_SIZE + i];
        scalar_t result;
        if (x > 20.0) {
            result = x;
        } else if (x < -20.0) {
            result = exp(x);
        } else {
            result = log1p(exp(x));
        }
        if (vec_idx + i < size) {
            output[vec_idx + i] = result;
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    constexpr int VEC_SIZE = 4;
    const int blocks = (size + (threads * VEC_SIZE) - 1) / (threads * VEC_SIZE);
    const size_t shared_mem = threads * VEC_SIZE * sizeof(typename torch::scalar_type<decltype(input)>::type);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_shared_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}