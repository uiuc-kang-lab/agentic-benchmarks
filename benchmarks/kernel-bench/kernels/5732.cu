#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel further optimizes by manually unrolling loops for common kernel sizes
// to reduce loop overhead and improve performance.

template <typename scalar_t>
__global__ void max_pool2d_unrolled_optimized_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    // Precompute base indices
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;

    // Manually unroll loops for common kernel sizes
    if (kernel_size == 2) {
        int ih0 = base_ih;
        int ih1 = base_ih + dilation;
        int iw0 = base_iw;
        int iw1 = base_iw + dilation;
        if (ih0 >= 0 && ih0 < input_height) {
            if (iw0 >= 0 && iw0 < input_width) max_val = max(max_val, input_channel[ih0 * input_width + iw0]);
            if (iw1 >= 0 && iw1 < input_width) max_val = max(max_val, input_channel[ih0 * input_width + iw1]);
        }
        if (ih1 >= 0 && ih1 < input_height) {
            if (iw0 >= 0 && iw0 < input_width) max_val = max(max_val, input_channel[ih1 * input_width + iw0]);
            if (iw1 >= 0 && iw1 < input_width) max_val = max(max_val, input_channel[ih1 * input_width + iw1]);
        }
    }
    else if (kernel_size == 3) {
        int ih0 = base_ih;
        int ih1 = base_ih + dilation;
        int ih2 = base_ih + 2 * dilation;
        int iw0 = base_iw;
        int iw1 = base_iw + dilation;
        int iw2 = base_iw + 2 * dilation;
        if (ih0 >= 0 && ih0 < input_height) {
            if (iw0 >= 0 && iw0 < input_width) max_val = max(max_val, input_channel[ih0 * input_width + iw0]);
            if (iw1 >= 0 && iw1 < input_width) max_val = max(max_val, input_channel[ih0 * input_width + iw1]);
            if (iw2 >= 0 && iw2 < input_width) max_val = max(max_val, input_channel[ih0 * input_width + iw2]);
        }
        if (ih1 >= 0 && ih1 < input_height) {
            if (iw0 >= 0 && iw0 < input_width) max_val = max(max_val, input_channel[ih1 * input_width + iw0]);
            if (iw1 >= 0 && iw1 < input_width) max_val = max(max_val, input_channel[ih1 * input_width + iw1]);
            if (iw2 >= 0 && iw2 < input_width) max_val = max(max_val, input_channel[ih1 * input_width + iw2]);
        }
        if (ih2 >= 0 && ih2 < input_height) {
            if (iw0 >= 0 && iw0 < input_width) max_val = max(max_val, input_channel[ih2 * input_width + iw0]);
            if (iw1 >= 0 && iw1 < input_width) max_val = max(max_val, input_channel[ih2 * input_width + iw1]);
            if (iw2 >= 0 && iw2 < input_width) max_val = max(max_val, input_channel[ih2 * input_width + iw2]);
        }
    }
    else {
        #pragma unroll 4
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = base_ih + kh * dilation;
            #pragma unroll 4
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = base_iw + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    max_val = max(max_val, input_channel[ih * input_width + iw]);
                }
            }
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_unrolled_optimized_kernel<scalar_t><<<blocks, threads>>>(
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
