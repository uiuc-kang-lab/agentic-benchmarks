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
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / warpSize;
    const unsigned int lane = tid % warpSize;
    const unsigned int warps_per_block = blockDim.x / warpSize;
    
    // Global index calculation
    int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread using grid-stride loop
    while (gid < numel) {
        // Load data into shared memory
        scalar_t val = __ldg(&x[gid]);
        
        // Process within warp first
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            // Use warp shuffle to share data
            scalar_t other = __shfl_down_sync(0xffffffff, val, offset);
            val = val < min_val ? min_val : (val > max_val ? max_val : val);
        }
        
        // Store to shared memory
        shared_data[tid] = val;
        __syncthreads();
        
        // Final processing and store
        if (tid < warpSize) {
            val = shared_data[tid];
            val = val < min_val ? min_val : (val > max_val ? max_val : val);
        }
        
        // Write result to global memory
        if (gid < numel) {
            out[gid] = val;
        }
        
        // Move to next chunk
        gid += stride;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));
    const int shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
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