#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that minimizes warp divergence by refactoring conditional logic
__global__ void branchless_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    // Flattened thread index over total output elements
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * in_channels * output_length;
    if (idx >= total_elements) return;

    // Map flattened index to (batch, channel, output position)
    const int o = idx % output_length;
    const int channel = (idx / output_length) % in_channels;
    const int batch = idx / (output_length * in_channels);

    // Compute base pointer for the current (batch, channel)
    const int input_base = batch * in_channels * input_length + channel * input_length;
    // Compute the starting index in the input corresponding to the current output element
    int start_idx = o * stride - padding;

    float sum = 0.0f;

    // If the pooling window is fully within bounds, we can avoid per-iteration checks
    if (start_idx >= 0 && start_idx + kernel_size <= input_length) {
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            sum += input[input_base + start_idx + k];
        }
    } else {
        // For boundary conditions, use an inline branchless check with unsigned comparison
        // Negative indices become large when cast to unsigned, guaranteeing the check fails
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start_idx + k;
            sum += ((unsigned)pos < (unsigned)input_length ? input[input_base + pos] : 0.0f);
        }
    }

    // Write the averaged result
    output[idx] = sum / kernel_size;
}

// Host function to launch the CUDA kernel
torch::Tensor branchless_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());
    int total = batch_size * in_channels * output_length;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    branchless_avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &branchless_avg_pool1d_forward, "Branchless 1D Average Pooling forward (CUDA)");
}
