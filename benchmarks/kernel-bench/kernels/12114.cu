#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int MAX_SIZE = 1024;

__constant__ float constant_targets[MAX_SIZE];

// Kernel uses constant memory for the 'targets' array
__global__ void hinge_loss_kernel_constant(const float* predictions, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = fmaxf(0.0f, 1.0f - predictions[i] * constant_targets[i]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();
    torch::Tensor output = torch::empty_like(predictions);

    // Optimize thread and block configuration
    int threads = 256;  // Keep 256 threads per block for good occupancy
    int max_blocks = 65535;  // Maximum blocks per grid dimension
    // Calculate optimal number of blocks based on SM count and data size
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    int sm_count = props.multiProcessorCount;
    // Aim for at least 2 blocks per SM for better occupancy
    int min_blocks_per_sm = 2;
    int target_blocks = sm_count * min_blocks_per_sm;
    // Ensure we have enough blocks to cover the data
    int blocks = min(max_blocks, max((n + threads - 1) / threads, target_blocks));

    // Copy targets to constant memory
    cudaMemcpyToSymbol(constant_targets, targets.data_ptr<float>(), n * sizeof(float));

    hinge_loss_kernel_constant<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Compute the mean of the output tensor
    auto mean = torch::mean(output);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward with constant memory optimization");
}
