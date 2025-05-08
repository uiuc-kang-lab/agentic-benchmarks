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
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_outputs = batch_size * channels * output_height * output_width;
    if (tidx >= total_outputs) return;

    const int ow = tidx % output_width;
    const int oh = (tidx / output_width) % output_height;
    const int c = (tidx / (output_width * output_height)) % channels;
    const int b = tidx / (output_width * output_height * channels);

    const int iw_start = ow * stride - padding;
    const int ih_start = oh * stride - padding;
    
    const int h_start = max(0, (ih_start + dilation - 1) / dilation);
    const int w_start = max(0, (iw_start + dilation - 1) / dilation);
    const int h_end = min(kernel_size, (input_height - ih_start + dilation - 1) / dilation);
    const int w_end = min(kernel_size, (input_width - iw_start + dilation - 1) / dilation);

    const int base_offset = b * channels * input_height * input_width + 
                           c * input_height * input_width;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = h_start; kh < h_end; ++kh) {
        const int ih = ih_start + kh * dilation;
        const int row_offset = ih * input_width;
        
        #pragma unroll
        for (int kw = w_start; kw < w_end; ++kw) {
            const int iw = iw_start + kw * dilation;
            const int input_idx = base_offset + row_offset + iw;
            max_val = max(max_val, __ldg(&input[input_idx]));
        }
    }

    output[tidx] = max_val;
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
    const int total_elements = batch_size * channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
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