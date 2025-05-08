#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses grid-stride loops to handle workloads larger than the number of active threads.
// Each thread processes multiple output elements by iterating through a flattened output index space.

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {

    // Total number of output elements in 4D tensor: (N, out_channels, output_height, output_width)
    int total = batch_size * out_channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    // Grid-stride loop: Each thread processes multiple elements if necessary
    for (int index = idx; index < total; index += step) {
        int temp = index;
        int ox = temp % output_width;
        temp /= output_width;
        int oy = temp % output_height;
        temp /= output_height;
        int oc = temp % out_channels;
        int n = temp / out_channels;

        float sum = 0.0f;
        // Convolution accumulation
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                int in_y = oy * stride - padding + kh;
                if (in_y < 0 || in_y >= input_height)
                    continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_x = ox * stride - padding + kw;
                    if (in_x < 0 || in_x >= input_width)
                        continue;
                    int input_idx = ((n * in_channels + ic) * input_height + in_y) * input_width + in_x;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        int output_idx = ((n * out_channels + oc) * output_height + oy) * output_width + ox;
        output[output_idx] = sum;
    }
}


// Host function to launch the kernel
// This implementation supports square kernels and does not support dilation or groups other than 1.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(dilation == 1, "Dilation other than 1 is not supported in this kernel");
    TORCH_CHECK(groups == 1, "Groups other than 1 are not supported in this kernel");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assuming square kernel

    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, x.options());
    
    int total_elements = batch_size * out_channels * output_height * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width,
        stride,
        padding);

    cudaDeviceSynchronize();

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using grid-stride loops");
}
