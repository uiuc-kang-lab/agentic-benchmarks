#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void cumsum_multi_stream_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int outer_size,
                                           int inner_size,
                                           int stride,
                                           int start_outer) {
    int outer_idx = start_outer + blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size) {
        float sum = 0.0f;
        for (int i = 0; i < stride; ++i) {
            int idx = outer_idx * stride * inner_size + i * inner_size + inner_idx;
            sum += __ldg(&input[idx]);
            output[idx] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x.size(i);
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= x.size(i);

    const int stride = x.size(dim);
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    ~for (int s = 0; s < num_streams; ++s) cudaStreamCreate(&streams[s]);~

    div_t chunk = div(outer_size, num_streams);
    for (int s = 0; s < num_streams; ++s) {
        int start = s * chunk.quot + min(s, chunk.rem);
        int end = start + chunk.quot + (s < chunk.rem);
        if (start >= outer_size) break;
        
        cumsum_multi_stream_kernel<<<end - start, inner_size, 0, streams[s]>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            outer_size,
            inner_size,
            stride,
            start
        );
    }

    ~for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }~

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Multi-stream CUDA cumulative sum");
}