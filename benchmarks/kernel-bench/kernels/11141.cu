#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function using __ldg() to load from read-only memory
__device__ float compute_max_logit_ldg(const float* __restrict__ logits, int num_classes) {
    float max_val = __ldg(&logits[0]);
    #pragma unroll 4
    for (int j = 1; j < num_classes; j++) {
        float logit = __ldg(&logits[j]);
        max_val = fmaxf(max_val, logit);
    }
    return max_val;
}

__device__ float compute_sum_exp_ldg(const float* __restrict__ logits, float max_val, int num_classes) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < num_classes; j++) {
        float logit = __ldg(&logits[j]);
        sum += expf(logit - max_val);
    }
    return sum;
}

__global__ void ce_loss_ldg_kernel(
    const float* __restrict__ logits,  // predictions, assumed to be aligned and read-only
    const int64_t* __restrict__ targets,
    float* losses,
    int batch_size,
    int num_classes
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Loop over samples with stride
    for (int i = idx; i < batch_size; i += totalThreads) {
        // Pointer to the logits for sample i; assume contiguous memory with 128-bit alignment
        const float* sample_logits = logits + i * num_classes;
        int target = __ldg(&targets[i]);

        // Use __ldg() in device functions for read-only loads
        float max_logit = compute_max_logit_ldg(sample_logits, num_classes);
        float sum_exp = compute_sum_exp_ldg(sample_logits, max_logit, num_classes);
        float log_sum_exp = logf(sum_exp);

        // Compute loss: - (logit_target - max_logit - log(sum_exp))
        losses[i] = -(__ldg(&sample_logits[target]) - max_logit - log_sum_exp);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.dim() == 2, "predictions must be a 2D tensor");
    TORCH_CHECK(targets.dim() == 1, "targets must be a 1D tensor");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "predictions must be Float32 tensor");
    TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be Int64 tensor");

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    TORCH_CHECK(targets.size(0) == batch_size, "targets must have same batch size as predictions");

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch kernel with typical configuration
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    ce_loss_ldg_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Error in ce_loss_ldg_kernel: ", cudaGetErrorString(err));

    return losses.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Cross Entropy Loss forward with __ldg for aligned memory (CUDA)");
}
