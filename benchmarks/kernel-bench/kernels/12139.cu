#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < n; i += stride) {
        output[i] = fmaxf(0.0f, __fmaf_rn(-predictions[i], targets[i], 1.0f));
    }
}

void hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets, torch::Tensor output, cudaStream_t stream) {
    int n = predictions.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hinge_loss_kernel<<<blocks, threads, 0, stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    hinge_loss_cuda(predictions, targets, output, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Streams");
}