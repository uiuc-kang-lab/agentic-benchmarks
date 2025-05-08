#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel using grid-stride loops to handle workloads larger than available threads
// and ensure correct boundary handling

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_stride_optim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int total_elements = batch_size * channels * output_height * output_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockDim.x * gridDim.x;

    while (index < total_elements) {
        int ow = index % output_width;
        int oh = (index / output_width) % output_height;
        int c = (index / (output_width * output_height)) % channels;
        int b = index / (output_width * output_height * channels);

        const int input_base = b * channels * input_height * input_width + 
                              c * input_height * input_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = oh * stride - padding + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int iw = ow * stride - padding + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = fmaxf(max_val, __ldg(&input[input_base + ih * input_width + iw]));
                    }
                }
            }
        }

        output[index] = max_val;
        index += gridSize;
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_stride_optim_kernel<scalar_t, 2><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
                break;
            case 3:
                max_pool2d_stride_optim_kernel<scalar_t, 3><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
                break;
            default:
                max_pool2d_stride_optim_kernel<scalar_t, -1><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Stride-optimized Max Pool 2D forward (CUDA)");
}
