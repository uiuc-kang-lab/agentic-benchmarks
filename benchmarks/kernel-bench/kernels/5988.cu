#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function that computes the sum over the pooling window
__device__ __forceinline__ float compute_avg_value(
    const float* __restrict__ input,
    const int input_offset,
    const int start_idx,
    const int kernel_size,
    const int input_length) {

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos = start_idx + k;
        if (pos >= 0 && pos < input_length) {
            sum += input[input_offset + pos];
        }
    }
    return sum;
}

// Main kernel for 1D average pooling using the modular device function
__global__ void modular_func_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * output_length;
    if (idx >= total_elements) return;

    // Compute output indices from flat index
    int o = idx % output_length;
    int channel = (idx / output_length) % in_channels;
    int batch = idx / (output_length * in_channels);

    int input_offset = batch * in_channels * input_length + channel * input_length;
    int start_idx = o * stride - padding;

    float sum = compute_avg_value(input, input_offset, start_idx, kernel_size, input_length);
    output[idx] = sum / kernel_size;
}

// Host function that wraps the kernel launch
torch::Tensor modular_func_avg_pool1d_forward(
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

    int total_elements = batch_size * in_channels * output_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    modular_func_avg_pool1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &modular_func_avg_pool1d_forward, "Modular Function 1D Average Pooling forward (CUDA)");
}
