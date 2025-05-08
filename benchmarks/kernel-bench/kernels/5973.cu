#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void block_tuned_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int input_length,
    const int output_length,
    const int batch_size,
    const int in_channels,
    const int block_size) {

    const int o = blockIdx.x * block_size + threadIdx.x;
    const int channel = blockIdx.y;
    const int batch = blockIdx.z;

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    const int input_offset = batch * in_channels * input_length + channel * input_length;
    const float scale = 1.0f / kernel_size;
    
    float sum = 0.0f;
    const int start_idx = o * stride - padding;
    
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        const int pos_input = start_idx + k;
        if (pos_input >= 0 && pos_input < input_length) {
            sum += input[input_offset + pos_input];
        }
    }
    
    output[batch * in_channels * output_length + channel * output_length + o] = sum * scale;
}

torch::Tensor block_tuned_avg_pool1d_forward(
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

    int block_size;
    if (output_length <= 128) block_size = 32;
    else if (output_length <= 256) block_size = 64;
    else if (output_length <= 512) block_size = 128;
    else block_size = 256;

    dim3 blocks((output_length + block_size - 1) / block_size, in_channels, batch_size);
    dim3 threads(block_size);

    block_tuned_avg_pool1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        batch_size,
        in_channels,
        block_size
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_tuned_avg_pool1d_forward, "Block tuned 1D Average Pooling forward (CUDA)");
}