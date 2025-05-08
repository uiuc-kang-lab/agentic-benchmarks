#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_CONST_SIZE 16384

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory buffers for predictions and targets
__constant__ float const_predictions[MAX_CONST_SIZE];
__constant__ float const_targets[MAX_CONST_SIZE];

// Kernel that uses constant memory for read-only predictions and targets
__global__ void hinge_loss_const_kernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < n; idx += stride) {
        float prod = const_predictions[idx] * const_targets[idx];
        float diff = 1.0f - prod;
        output[idx] = diff > 0.0f ? diff : 0.0f;
    }
}

// Fallback kernel that uses global memory if data size exceeds constant memory limits
__global__ void hinge_loss_global_kernel(const float* predictions, const float* targets, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < n; idx += stride) {
        float prod = predictions[idx] * targets[idx];
        float diff = 1.0f - prod;
        output[idx] = diff > 0.0f ? diff : 0.0f;
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // Use constant memory if the data fits within hardware limits
    if(n <= MAX_CONST_SIZE) {
        // Copy predictions and targets into constant memory. The copy type is DeviceToDevice since tensors are already on GPU.
        cudaMemcpyToSymbol(const_predictions, predictions.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(const_targets, targets.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        hinge_loss_const_kernel<<<blocks, threads>>>(output.data_ptr<float>(), n);
    } else {
        hinge_loss_global_kernel<<<blocks, threads>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), n);
    }

    // Compute and return the mean hinge loss
    return torch::mean(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward using Constant Memory");
}
