#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesced_memory_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * in_channels * output_length;

    if (idx >= total_elements) return;

    const int o = idx % output_length;
    const int channel = (idx / output_length) % in_channels;
    const int batch = idx / (output_length * in_channels);

    const int input_offset = batch * in_channels * input_length + channel * input_length;
    const float scale = 1.0f / kernel_size;
    
    float sum = 0.0f;
    const int start_idx = o * stride - padding;
    
    for (int k = 0; k < kernel_size; ++k) {
        const int pos_input = start_idx + k;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[input_offset + pos_input];
        }
    }
    
    output[idx] = sum * scale;
}

torch::Tensor coalesced_memory_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid parameters");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    const int total_elements = batch_size * in_channels * output_length;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    coalesced_memory_avg_pool1d_kernel<<<blocks, threads>>>(
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
    m.def("forward", &coalesced_memory_avg_pool1d_forward, "Coalesced Memory 1D Average Pooling forward (CUDA)");
}