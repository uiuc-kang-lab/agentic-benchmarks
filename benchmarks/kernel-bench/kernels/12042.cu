#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to compute the hinge loss for a single element
__device__ inline float compute_hinge_loss(float prediction, float target) {
    return fmaxf(0.0f, 1.0f - prediction * target);
}

// Kernel that uses a grid-stride loop to process all elements
__global__ void hinge_loss_kernel(const float* __restrict__ predictions,
                                   const float* __restrict__ targets,
                                   float* __restrict__ output,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        output[i] = compute_hinge_loss(pred, targ);
    }
}

// PyTorch binding function
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    // Compute and return the mean hinge loss
    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Hinge Loss Forward");
}
