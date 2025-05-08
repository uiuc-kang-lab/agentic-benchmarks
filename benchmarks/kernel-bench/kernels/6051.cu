#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void avg_pool1d_kernel_uniform(
    const float *__restrict__ input,
    float *output,
    int kernel_size,
    int stride,
    int padding,
    int input_length,
    int output_length,
    int batch_size,
    int in_channels) {

    int channel = blockIdx.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;
    int o = blockIdx.x * blockDim.x + tid;

    if (channel >= in_channels || batch >= batch_size) return;

    if (o < output_length) {
        float sum = 0.0f;
        int pos_padded = o * stride + padding;
        int pos_input_start = pos_padded - padding;

        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            int pos_input = pos_input_start + k;
            float input_val = 0.0f;
            if (pos_input >= 0 && pos_input < input_length) {
                input_val = input[batch * in_channels * input_length + channel * input_length + pos_input];
            }
            sum += input_val;
        }

        output[batch * in_channels * output_length + channel * output_length + o] = sum / kernel_size;
    }
}

torch::Tensor avg_pool1d_forward_uniform(
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

    dim3 threads(BLOCK_SIZE);
    dim3 grid(
        (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        in_channels,
        batch_size
    );

    avg_pool1d_kernel_uniform<<<grid, threads>>>(
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
    m.def("forward", &avg_pool1d_forward_uniform, "1D Average Pooling with uniform control flow (CUDA)");
}
