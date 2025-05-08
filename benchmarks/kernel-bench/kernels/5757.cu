#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

template <typename scalar_t>
__device__ __forceinline__ int get_input_index(
    int b, int c, int h, int w,
    int channels, int height, int width
) {
    return ((b * channels + c) * height + h) * width + w;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max(
    const scalar_t* __restrict__ input,
    int b, int c,
    int oh, int ow,
    int input_h, int input_w,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = oh * stride - padding + kh * dilation;
        if (ih < 0 || ih >= input_h) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int iw = ow * stride - padding + kw * dilation;
            if (iw >= 0 && iw < input_w) {
                const int idx = get_input_index<scalar_t>(b, c, ih, iw, 0, input_h, input_w);
                max_val = max(max_val, input[idx]);
            }
        }
    }
    return max_val;
}

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_h,
    const int input_w,
    const int output_h,
    const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels * output_h * output_w) return;

    const int ow = index % output_w;
    const int oh = (index / output_w) % output_h;
    const int c = (index / (output_w * output_h)) % channels;
    const int b = index / (output_w * output_h * channels);

    output[index] = compute_max<scalar_t>(
        input, b, c, oh, ow,
        input_h, input_w,
        kernel_size, stride,
        padding, dilation
    );
}

} // anonymous namespace

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Dimension calculations with explicit casting
    const int64_t input_h = input.size(2);
    const int64_t input_w = input.size(3);
    
    const int output_h = static_cast<int>(
        (input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    );
    const int output_w = static_cast<int>(
        (input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    );

    auto output = torch::empty({input.size(0), input.size(1), output_h, output_w}, input.options());

    const int threads = 256;
    const int elements = output.numel();
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.size(0),
            input.size(1),
            input_h,
            input_w,
            output_h,
            output_w,
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
