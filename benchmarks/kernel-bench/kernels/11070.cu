#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_loss_stride_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Stride loop for handling large workloads
    for (int i = tid; i < batch_size; i += stride) {
        const float* logits_i = logits + i * num_classes;
        int64_t target = targets[i];

        float max_logit = logits_i[0];
        for (int j = 1; j < num_classes; ++j) {
            max_logit = fmaxf(max_logit, logits_i[j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(logits_i[j] - max_logit);
        }

        float log_sum_exp = logf(sum_exp);
        losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_loss_stride_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CrossEntropyLoss with stride loop optimization (CUDA)");
}