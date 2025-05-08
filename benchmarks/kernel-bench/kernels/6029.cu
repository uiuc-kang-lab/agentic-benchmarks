#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Flattened kernel: each thread computes one output element based on flattened index
__global__ void flat_avg_pool1d_kernel(
    const float *input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int in_channels,
    int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * output_length;
    if (idx >= total) return;

    // Determine batch, channel, and output spatial position from flattened index
    int out_per_batch = in_channels * output_length;
    int batch = idx / out_per_batch;
    int rem = idx % out_per_batch;
    int channel = rem / output_length;
    int o = rem % output_length;

    // Compute the starting position of the pooling window (with padding)
    int base_index = batch * (in_channels * input_length) + channel * input_length;
    int window_start = o * stride - padding;
    float sum = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int ipos = window_start + k;
        if (ipos >= 0 && ipos < input_length) {
            int input_index = batch * (in_channels * input_length) + channel * input_length + ipos;
            sum += input[input_index];
        }
    }
    
    output[idx] = sum / kernel_size;
}

torch::Tensor avg_pool1d_forward(
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
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    flat_avg_pool1d_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kernel_size,
        stride,
        padding,
        input_length,
        output_length,
        in_channels,
        batch_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward (CUDA)");
}
