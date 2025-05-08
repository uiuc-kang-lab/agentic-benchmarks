#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int idx = blockIdx.x * block_size + tid;
    const int grid_stride = gridDim.x * block_size;
    
    while (idx < size) {
        // Load data into shared memory
        shared_data[tid] = (idx < size) ? input[idx] : 0;
        __syncthreads();  // Ensure shared memory is loaded
        
        // Process data from shared memory
        if (idx < size) {
            output[idx] = compute_softplus(shared_data[tid]);
        }
        
        idx += grid_stride;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const int shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}