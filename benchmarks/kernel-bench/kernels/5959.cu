#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 1D average pooling with manual loop unrolling
__global__ void manual_unroll_avg_pool1d_kernel(
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

    if (o >= output_length || channel >= in_channels || batch >= batch_size) return;

    float sum = 0.0f;

    // Manually unroll the loop over the kernel window
    if (kernel_size == 3) {
        for (int k = 0; k < 3; ++k) {
            int pos_padded = o * stride + k;
            int pos_input = pos_padded - padding;
            if (pos_input >= 0 && pos_input < input_length) {
                int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
                sum += input[input_idx];
            }
        }
    } else if (kernel_size == 5) {
        for (int k = 0; k < 5; ++k) {
            int pos_padded = o * stride + k;
            int pos_input = pos_padded - padding;
            if (pos_input >= 0 && pos_input < input_length) {
                int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
                sum += input[input_idx];
            }
        }
    } else {
        for (int k = 0; k < kernel_size; ++k) {
            int pos_padded = o * stride + k;
            int pos_input = pos_padded - padding;
            if (pos_input >= 0 && pos_input < input_length) {
                int input_idx = batch * in_channels * input_length + channel * input_length + pos_input;
                sum += input[input_idx];
            }
        }
    }

    int output_idx = batch * in_channels * output_length + channel * output_length + o;
    output[output_idx] = sum / kernel_size;
}

// Host function to launch the CUDA kernel
torch::Tensor manual_unroll_avg_pool1d_forward(
    const torch::Tensor &x,
    int kernel_size,
    int stride,
    int padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be a 3D tensor");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0, "Invalid kernel parameters");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_length}, x.options());

    dim3 threads(256);
    dim3 grid(
        (output_length + threads.x - 1) / threads.x,
        in_channels,
        batch_size
    );

    manual_unroll_avg_pool1d_kernel<<<grid, threads>>>(
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
    m.def("forward", &manual_unroll_avg_pool1d_forward, "1D Average Pooling forward (CUDA) with manual loop unrolling");
}
