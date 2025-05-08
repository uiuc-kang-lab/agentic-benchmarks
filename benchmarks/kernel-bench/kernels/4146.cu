#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

// Kernel to apply HardTanh activation with streaming

__global__ void hardtanh_kernel_stream(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int64_t numel,
                                       float min_val,
                                       float max_val,
                                       int stream_offset) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x + stream_offset;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        float val = __ldg(&x[i]);
        out[i] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

at::Tensor forward_cuda_stream(const at::Tensor& x, float min_val, float max_val) {
    auto out = at::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    int stream_chunk = numel / 4;

    for (int i = 0; i < 4; ++i) { // Using 4 streams
        int offset = i * stream_chunk;
        hardtanh_kernel_stream<<<blocks, threads, 0, stream.stream()>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            stream_chunk,
            min_val,
            max_val,
            offset);
    }

    cudaStreamSynchronize(stream.stream());
    return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
    if (!x.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor");
    }
    return forward_cuda_stream(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardTanh activation with CUDA streams");
}
