#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel optimized to distribute workload evenly across blocks and threads
__global__ void max_pool2d_kernel(float* input, float* output,
                                  int batch_size, int channels, 
                                  int input_height, int input_width,
                                  int output_height, int output_width, 
                                  int kernel_size, int stride,
                                  int padding, int dilation,
                                  int elems_per_thread)
{
    // Calculate the global index for each thread
    const int total_outputs = batch_size * channels * output_height * output_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int element_idx = 0; element_idx < elems_per_thread; element_idx++) {
        int output_idx = idx + element_idx * (gridDim.x * blockDim.x);
        if (output_idx >= total_outputs) return;
        const int ow = output_idx % output_width;
        const int oh = (output_idx / output_width) % output_height;
        const int c = (output_idx / (output_width * output_height)) % channels;
        const int b = output_idx / (output_width * output_height * channels);

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
        output[output_idx] = max_val;
    }
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const auto output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int total_output = batch_size * channels * output_height * output_width;
    const int blocks = (total_output + threads - 1) / threads;
    // Calculate how many elements each thread should handle
    const int elems_per_thread = (total_output + (blocks * threads) - 1) / (blocks * threads);

    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation,
        elems_per_thread
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward optimized with workload distribution (CUDA)");
}
