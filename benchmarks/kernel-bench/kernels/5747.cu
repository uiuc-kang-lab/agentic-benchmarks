#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int MAX_KERNEL_SIZE = 7;

template <int KERNEL_SIZE, typename scalar_t>
__device__ void process_kernel(
    const scalar_t* __restrict__ input_channel,
    scalar_t& max_val,
    int base_ih,
    int base_iw,
    int dilation,
    int input_height,
    int input_width
) {
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int ih = base_ih + kh * dilation;
        const bool valid_h = ih >= 0 && ih < input_height;
        
        #pragma unroll
        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
            const int iw = base_iw + kw * dilation;
            if (valid_h && iw >= 0 && iw < input_width) {
                max_val = max(max_val, __ldg(&input_channel[ih * input_width + iw]));
            }
        }
    }
}

template <typename scalar_t>
__device__ void process_kernel_generic(
    const scalar_t* __restrict__ input_channel,
    scalar_t& max_val,
    int base_ih,
    int base_iw,
    int dilation,
    int input_height,
    int input_width,
    int kernel_size
) {
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = base_ih + kh * dilation;
        const bool valid_h = ih >= 0 && ih < input_height;
        
        #pragma unroll 4
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = base_iw + kw * dilation;
            if (valid_h && iw >= 0 && iw < input_width) {
                max_val = max(max_val, __ldg(&input_channel[ih * input_width + iw]));
            }
        }
    }
}

template <typename scalar_t>
__global__ void max_pool2d_textured_kernel(
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
    const int ow = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (oh >= output_height) return;

    scalar_t max_val[2] = {-std::numeric_limits<scalar_t>::infinity(),
                          -std::numeric_limits<scalar_t>::infinity()};
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    for (int i = 0; i < 2; ++i) {
        const int curr_ow = ow + i * blockDim.x;
        if (curr_ow >= output_width) break;

        const int base_ih = oh * stride - padding;
        const int base_iw = curr_ow * stride - padding;

        switch (kernel_size) {
            case 2: process_kernel<2>(input_channel, max_val[i], base_ih, base_iw,
                                     dilation, input_height, input_width); break;
            case 3: process_kernel<3>(input_channel, max_val[i], base_ih, base_iw,
                                     dilation, input_height, input_width); break;
            default:
                if (kernel_size <= MAX_KERNEL_SIZE) {
                    process_kernel_generic(input_channel, max_val[i], base_ih, base_iw,
                                          dilation, input_height, input_width, kernel_size);
                }
        }

        if (curr_ow < output_width) {
            output[(b * channels + c) * output_height * output_width + oh * output_width + curr_ow] = max_val[i];
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

    const dim3 threads(32, 4);  // 2x wider blocks for better memory coalescing
    const dim3 blocks(
        (output_width + (threads.x * 2) - 1) / (threads.x * 2),
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_textured_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Textured Max Pool 2D forward (CUDA)");
}