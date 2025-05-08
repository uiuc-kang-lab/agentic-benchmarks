#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int STRIDE>
__global__ void hinge_loss_kernel(const float* __restrict__ predictions,
                                 const float* __restrict__ targets,
                                 float* __restrict__ output,
                                 int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gid = tid * STRIDE;

    #pragma unroll
    for(int i = 0; i < STRIDE; ++i) {
        int idx = gid + i;
        if(idx < n) {
            float val = 1.0f - predictions[idx] * targets[idx];
            output[idx] = val > 0.0f ? val : 0.0f;
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    const int VECTOR_SIZE = 4;
    const int threads = 256;
    const int elements_per_block = threads * VECTOR_SIZE;
    int blocks = (n + elements_per_block - 1) / elements_per_block;
    blocks = min(blocks, 65535);

    hinge_loss_kernel<VECTOR_SIZE><<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Hinge Loss Forward");
}