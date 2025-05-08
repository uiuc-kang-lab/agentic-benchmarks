#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

template <typename scalar_t>
__global__ void hardtanh_kernel(const scalar_t* __restrict__ x,
                               scalar_t* __restrict__ out,
                               int64_t numel,
                               scalar_t min_val,
                               scalar_t max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = x[idx];
        val = fmaxf(val, min_val);
        val = fminf(val, max_val);
        out[idx] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();
    
    // Create multiple CUDA streams
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 512;
    const int64_t elements_per_stream = (numel + num_streams - 1) / num_streams;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int64_t stream_start = i * elements_per_stream;
            const int64_t stream_elements = std::min(elements_per_stream, 
                                                   numel - stream_start);
            if (stream_elements <= 0) break;
            
            const int stream_blocks = (stream_elements + threads - 1) / threads;
            
            hardtanh_kernel<scalar_t><<<stream_blocks, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>() + stream_start,
                out.data_ptr<scalar_t>() + stream_start,
                stream_elements,
                static_cast<scalar_t>(min_val),
                static_cast<scalar_t>(max_val)
            );
        }
    }));
    
    // Synchronize and cleanup streams
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
    m.def("forward", &forward, "HardTanh activation with streams (CUDA)");
}