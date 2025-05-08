#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel(const float* x, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = x[idx] / (1.0f + fabsf(x[idx]));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *d_x, *d_out;
    cudaMalloc(&d_x, num_elements * sizeof(float));
    cudaMalloc(&d_out, num_elements * sizeof(float));

    cudaMemcpyAsync(d_x, x.data_ptr<float>(), num_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    softsign_kernel<<<blocks, threads, 0, stream>>>(d_x, d_out, num_elements);

    cudaMemcpyAsync(out.data_ptr<float>(), d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_x);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with streams (CUDA)");
}