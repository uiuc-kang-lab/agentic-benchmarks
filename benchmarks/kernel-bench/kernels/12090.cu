#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_kernel_stream(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float pred = __ldg(&predictions[i]);
    float targ = __ldg(&targets[i]);
    float diff = 1.0f - pred * targ;
    output[i] = diff > 0.0f ? diff : 0.0f;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Create a CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the kernel asynchronously in the specified stream
    hinge_loss_kernel_stream<<<blocks, threads, 0, stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Record calculation completion to allow overlap of subsequent operations
    cudaEvent_t kernel_done;
    cudaEventCreate(&kernel_done);
    cudaEventRecord(kernel_done, stream);

    // Wait for the stream to complete the tasks (kernel and event)
    cudaStreamSynchronize(stream);

    // Destroy the stream
    cudaStreamDestroy(stream);

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Streams");
}