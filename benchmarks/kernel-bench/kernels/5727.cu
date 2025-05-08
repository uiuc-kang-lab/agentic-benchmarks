#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_uniform_kernel(
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
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    // Pre-calculate input base indices and valid ranges
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;
    
    // Calculate valid ranges once
    const int h_start = max(0, (base_ih < 0) ? (-base_ih + dilation - 1) / dilation : 0);
    const int w_start = max(0, (base_iw < 0) ? (-base_iw + dilation - 1) / dilation : 0);
    const int h_end = min(KERNEL_SIZE, (input_height - base_ih + dilation - 1) / dilation);
    const int w_end = min(KERNEL_SIZE, (input_width - base_iw + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    // Uniform flow without branches in the inner loop
    #pragma unroll
    for (int kh = h_start; kh < h_end; kh++) {
        const int ih = base_ih + kh * dilation;
        const int row_offset = ih * input_width;
        
        #pragma unroll
        for (int kw = w_start; kw < w_end; kw++) {
            const int iw = base_iw + kw * dilation;
            max_val = max(max_val, input_channel[row_offset + iw]);
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
}

template<typename scalar_t>
__global__ void max_pool2d_uniform_kernel_dynamic(
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

    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;
    
    // Pre-calculate valid ranges
    const int h_start = max(0, (base_ih < 0) ? (-base_ih + dilation - 1) / dilation : 0);
    const int w_start = max(0, (base_iw < 0) ? (-base_iw + dilation - 1) / dilation : 0);
    const int h_end = min(kernel_size, (input_height - base_ih + dilation - 1) / dilation);
    const int w_end = min(kernel_size, (input_width - base_iw + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;

    #pragma unroll 4
    for (int kh = h_start; kh < h_end; kh++) {
        const int ih = base_ih + kh * dilation;
        const int row_offset = ih * input_width;
        
        #pragma unroll 4
        for (int kw = w_start; kw < w_end; kw++) {
            const int iw = base_iw + kw * dilation;
            max_val = max(max_val, input_channel[row_offset + iw]);
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
        if (kernel_size == 2) {
            max_pool2d_uniform_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
        else if (kernel_size == 3) {
            max_pool2d_uniform_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
        else {
            max_pool2d_uniform_kernel_dynamic<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                kernel_size, stride, padding, dilation
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}