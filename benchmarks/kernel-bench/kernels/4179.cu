#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_hardtanh_reduce(scalar_t val, scalar_t min_val, scalar_t max_val) {
    return max(min(val, max_val), min_val);
}

template <typename scalar_t>
__global__ void hardtanh_hybrid_kernel(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ out,
                                      int64_t numel,
                                      scalar_t min_val,
                                      scalar_t max_val) {
    __shared__ scalar_t s_min_val, s_max_val;
    s_min_val = min_val;
    s_max_val = max_val;
    if (threadIdx.x == 0) {
        s_min_val = min_val;
        s_max_val = max_val;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int64_t idx = tid; idx < numel; idx += stride) {
        scalar_t val = x[idx];
        val = warp_hardtanh_reduce(val, s_min_val, s_max_val);
        out[idx] = val;
    }
}

at::Tensor forward_cuda(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks = std::min(65535, (int)((numel + threads - 1) / threads));
    
    const int num_streams = (numel > 1048576) ? 4 : 1;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cuda", ([&] {
        const int64_t elements_per_stream = (numel + num_streams - 1) / num_streams;
        
        for (int i = 0; i < num_streams; i++) {
            const int64_t stream_start = i * elements_per_stream;
            const int64_t stream_elements = std::min(elements_per_stream, numel - stream_start);
            if (stream_elements <= 0) break;

            hardtanh_hybrid_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
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
    m.def("forward", &forward, "HardTanh hybrid optimization (CUDA)");
}