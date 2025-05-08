#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void hinge_loss_shared_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {
    __shared__ float spreds[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    __shared__ float stargets[BLOCK_SIZE * ELEMENTS_PER_THREAD];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + tid;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (global_idx + i * BLOCK_SIZE < n) {
            spreds[tid + i * BLOCK_SIZE] = predictions[global_idx + i * BLOCK_SIZE];
            stargets[tid + i * BLOCK_SIZE] = targets[global_idx + i * BLOCK_SIZE];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int local_idx = tid + i * BLOCK_SIZE;
        if (global_idx + i * BLOCK_SIZE < n) {
            float val = spreds[local_idx] * stargets[local_idx];
            output[global_idx + i * BLOCK_SIZE] = fmaxf(0.0f, 1.0f - val);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int blocks = (n + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    blocks = min(blocks, 65535);

    hinge_loss_shared_kernel<<<blocks, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with Shared Memory");
}