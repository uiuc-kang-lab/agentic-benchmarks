#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that computes hinge loss using a grid-stride loop and block-level reduction
__global__ void hinge_loss_stride_kernel(const float* __restrict__ predictions,
                                           const float* __restrict__ targets,
                                           float* partialSums,
                                           const int n) {
    extern __shared__ float sdata[];  // Dynamic shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Grid-stride loop to cover all elements and ensure boundary correctness
    for (int i = idx; i < n; i += stride) {
        float p = __ldg(&predictions[i]);
        float t = __ldg(&targets[i]);
        local_sum += fmaxf(0.0f, 1.0f - p * t);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // In-block reduction to sum the losses
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// PyTorch binding function
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto partialSums = torch::empty({blocks}, predictions.options());

    // Launch kernel with dynamic shared memory allocated for block reduction
    hinge_loss_stride_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partialSums.data_ptr<float>(),
        n
    );

    // Compute the final mean hinge loss from the block partial sums
    auto loss_sum = torch::sum(partialSums);
    auto mean_loss = loss_sum / n;
    return mean_loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride Loop Hinge Loss Forward");
}
