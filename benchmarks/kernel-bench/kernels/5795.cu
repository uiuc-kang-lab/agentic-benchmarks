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
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ox >= output_width || oy >= output_height) return;

    const int channels_per_thread = (channels + 31) / 32;
    
    for (int b = blockIdx.z; b < batch_size; b += gridDim.z) {
        for (int c_base = 0; c_base < channels; c_base += channels_per_thread) {
            for (int c_offset = 0; c_offset < channels_per_thread && (c_base + c_offset) < channels; c_offset++) {
                const int c = c_base + c_offset;
                
                scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    const int ih = oy * stride - padding + kh * dilation;
                    if (ih >= 0 && ih < input_height) {
                        #pragma unroll
                        for (int kw = 0; kw < kernel_size; kw++) {
                            const int iw = ox * stride - padding + kw * dilation;
                            if (iw >= 0 && iw < input_width) {
                                const int input_idx = b * (channels * input_height * input_width) +
                                                    c * (input_height * input_width) +
                                                    ih * input_width +
                                                    iw;
                                max_val = max(max_val, input[input_idx]);
                            }
                        }
                    }
                }

                const int output_idx = b * (channels * output_height * output_width) +
                                     c * (output_height * output_width) +
                                     oy * output_width +
                                     ox;
                output[output_idx] = max_val;
            }
        }
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        min(batch_size, 32)
    );

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