#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    int batch = blockIdx.z;

    if (channel >= in_channels || batch >= batch_size) return;

    for (int idx = o; idx < output_length; idx += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        int offset = idx * stride - padding;
        int start_k = offset < 0 ? -offset : 0;
        int end_k = (input_length - offset) < kernel_size ? (input_length - offset) : kernel_size;
        for (int k = start_k; k < end_k; ++k) {
            int pos_input = offset + k;
            int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
            sum += input[input_idx];
        }
        output[batch * in_channels * output_length + channel * output_length + idx] = sum / kernel_size;
    }
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

    // Experiment with different block sizes
    int blockSize = 128;  // Chosen based on typical warp size and balance
    dim3 threads(blockSize);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel<<<grid, threads>>>(
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
    m.def("forward", &avg_pool1d_forward, "1D Average Pooling forward with tuned block size (CUDA)");
}