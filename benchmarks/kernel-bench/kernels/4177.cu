#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_hardtanh_reduce(scalar_t val, scalar_t min_val, scalar_t max_val) {
    return max(min_val, min(val, max_val));
}

template <typename scalar_t>
__global__ void hardtanh_optimized_kernel(const scalar_t* __restrict__ x,
                                        scalar_t* __restrict__ out,
                                        int64_t numel,
                                        scalar_t min_val,
                                        scalar_t max_val) {
    using Vec4 = typename std::aligned_storage<sizeof(float4), alignof(float4)>::type;
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    const int64_t vec_size = 4;
    const int64_t vec_numel = numel / vec_size;
    
    for (int64_t i = idx; i < vec_numel; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(x)[i];
        float4 out_vec;
        
        out_vec.x = warp_hardtanh_reduce(in_vec.x, min_val, max_val);
        out_vec.y = warp_hardtanh_reduce(in_vec.y, min_val, max_val);
        out_vec.z = warp_hardtanh_reduce(in_vec.z, min_val, max_val);
        out_vec.w = warp_hardtanh_reduce(in_vec.w, min_val, max_val);
        
        reinterpret_cast<float4*>(out)[i] = out_vec;
    }
    
    for (int64_t i = vec_numel * vec_size + idx; i < numel; i += stride) {
        out[i] = warp_hardtanh_reduce(x[i], min_val, max_val);
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    const int num_streams = (numel > 1048576) ? 4 : 1;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    const int64_t elements_per_stream = (numel + num_streams - 1) / num_streams;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int64_t stream_start = i * elements_per_stream;
            const int64_t stream_elements = std::min(elements_per_stream, 
                                                   numel - stream_start);
            if (stream_elements <= 0) break;
            
            const int stream_blocks = (stream_elements + threads - 1) / threads;
            
            hardtanh_optimized_kernel<scalar_t><<<stream_blocks, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>() + stream_start,
                out.data_ptr<scalar_t>() + stream_start,
                stream_elements,
                static_cast<scalar_t>(min_val),
                static_cast<scalar_t>(max_val)
            );
        }
    }));
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh optimized hybrid implementation (CUDA)");
}