#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel leveraging shared memory for HardTanh activation
template <typename scalar_t>
__global__ void hardtanh_shared_kernel(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ out,
                                        int64_t numel,
                                        scalar_t min_val,
                                        scalar_t max_val) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int threadId = threadIdx.x;
    const int64_t globalId = blockIdx.x * blockDim.x + threadId;
    
    // Load global data into shared memory
    if (globalId < numel) {
        shared_data[threadId] = x[globalId];
    }
    __syncthreads();
    
    // Process the data in shared memory and apply HardTanh
    if (globalId < numel) {
        scalar_t val = shared_data[threadId];
        // Clamp the value between min_val and max_val
        val = (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
        shared_data[threadId] = val;
    }
    __syncthreads();

    // Write the results back to global memory
    if (globalId < numel) {
        out[globalId] = shared_data[threadId];
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    // Allocate shared memory: one element per thread
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_shared_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel,
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
        );
    }));
    
    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh activation with shared memory (CUDA)");
}
