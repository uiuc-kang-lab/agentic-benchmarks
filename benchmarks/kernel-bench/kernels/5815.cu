#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch = blockIdx.z;
    const int channel = blockIdx.y;
    const int oh = blockIdx.x * blockDim.x + threadIdx.x;
    const int ow = threadIdx.y;

    if (oh >= output_height || channel >= channels || batch >= batch_size) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = base_h + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int iw = base_w + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;

            const int input_idx = ((batch * channels + channel) * input_height + ih) * input_width + iw;
            max_val = max(max_val, input[input_idx]);
        }
    }

    if (ow < output_width) {
        const int output_idx = ((batch * channels + channel) * output_height + oh) * output_width + ow;
        atomicMax(&output[output_idx], max_val);
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::full({batch_size, channels, output_height, output_width}, -std::numeric_limits<float>::infinity(), input.options());

    dim3 threads(32, 4);
    dim3 blocks(
        (output_height + threads.x - 1) / threads.x,
        channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}