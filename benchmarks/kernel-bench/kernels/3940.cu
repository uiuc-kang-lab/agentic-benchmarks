#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_streamed(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    if (tid < num_elements) {
        float val = x[tid];
        out[tid] = val / (1.0f + fabsf(val));
    }
}

// Update the forward function to use CUDA streams
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *x_ptr = x.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();

    // Use cudaMemcpyAsync for overlapping memory operations
    cudaMemcpyAsync(out_ptr, x_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Launch the kernel with the created stream
    softsign_kernel_streamed<<<blocks, threads, 0, stream>>>(x_ptr, out_ptr, num_elements);

    // Synchronize the stream to ensure all operations are complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with CUDA streams (CUDA)");
}