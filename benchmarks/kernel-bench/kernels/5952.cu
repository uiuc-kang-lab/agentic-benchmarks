#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Templated kernel that uses BLOCK_SIZE as a compile-time parameter
template <unsigned int BLOCK_SIZE>
__global__ void avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    int o = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (o < output_length && channel < in_channels && batch < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int pos_padded = o * stride + k;
            int pos_input = pos_padded - padding;
            if (pos_input >= 0 && pos_input < input_length) {
                int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
                sum += input[input_idx];
            }
        }
        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
    }
}

// Forward function with an additional block_size parameter to experiment with configuration
torch::Tensor avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding,
    int block_size = 256) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    // Allow only specific block sizes to ensure compile-time optimization
    if (block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256 && block_size != 512) {
        block_size = 256; // default fallback
    }

    dim3 threads(block_size);
    dim3 grid((output_length + block_size - 1) / block_size, in_channels, batch_size);

    // Dispatch kernel based on the selected block_size
    if (block_size == 32) {
        avg_pool1d_kernel<32><<<grid, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels);
    } else if (block_size == 64) {
        avg_pool1d_kernel<64><<<grid, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels);
    } else if (block_size == 128) {
        avg_pool1d_kernel<128><<<grid, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels);
    } else if (block_size == 256) {
        avg_pool1d_kernel<256><<<grid, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels);
    } else if (block_size == 512) {
        avg_pool1d_kernel<512><<<grid, threads>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            kernel_size,
            stride,
            padding,
            input_length,
            output_length,
            batch_size,
            in_channels);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}
