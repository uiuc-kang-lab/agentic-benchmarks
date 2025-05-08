#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename scalar_t>
__global__ void hardtanh_kernel_shared(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ out,
                                      int64_t numel,
                                      scalar_t min_val,
                                      scalar_t max_val) {
    const int TILE_SIZE = 1024;
    __shared__ scalar_t shared_data[TILE_SIZE];
    
    int64_t tid = threadIdx.x;
    int64_t block_offset = blockIdx.x * TILE_SIZE;
    int64_t grid_stride = TILE_SIZE * gridDim.x;
    
    // Process multiple tiles per block
    for (int64_t base = block_offset; base < numel; base += grid_stride) {
        int64_t remaining = min(static_cast<int64_t>(TILE_SIZE), numel - base);
        
        // Cooperative load into shared memory
        if (tid < remaining) {
            shared_data[tid] = x[base + tid];
        }
        
        // Only synchronize if there are multiple threads accessing shared memory
        if (remaining > 32) {
            __syncthreads();
        }
        
        // Process data from shared memory
        if (tid < remaining) {
            scalar_t val = shared_data[tid];
            val = val < min_val ? min_val : (val > max_val ? max_val : val);
            out[base + tid] = val;
        }
        
        // Only synchronize if there are multiple threads accessing shared memory
        if (remaining > 32) {
            __syncthreads();
        }
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    const int threads = 1024;
    const int blocks = min(65535, (numel + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel_shared<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &forward, "HardTanh activation (CUDA)");
}