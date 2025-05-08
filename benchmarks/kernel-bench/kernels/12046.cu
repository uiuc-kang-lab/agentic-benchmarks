#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_adaptive_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float pred = __ldg(&predictions[idx]);
        const float target = __ldg(&targets[idx]);
        output[idx] = fmaxf(0.0f, 1.0f - pred * target);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Experiment with different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256; // Default block size
    int min_time = INT_MAX;

    for (int i = 0; i < 5; ++i) {
        int threads = block_sizes[i];
        int blocks = (n + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        hinge_loss_adaptive_kernel<<<blocks, threads>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds < min_time) {
            min_time = milliseconds;
            optimal_block_size = threads;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Launch kernel with optimal block size
    int blocks = (n + optimal_block_size - 1) / optimal_block_size;
    hinge_loss_adaptive_kernel<<<blocks, optimal_block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Block Hinge Loss Forward");
}