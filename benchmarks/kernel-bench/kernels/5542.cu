#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int KERNEL_PARAMS[8];  // input_height, input_width, output_height, output_width, kernel_size, stride, padding, dilation

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels
) {
    // Block handles 32x32 output elements
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Get input/output dimensions from constant memory
    const int input_height = KERNEL_PARAMS[0];
    const int input_width = KERNEL_PARAMS[1];
    const int output_height = KERNEL_PARAMS[2];
    const int output_width = KERNEL_PARAMS[3];
    const int kernel_size = KERNEL_PARAMS[4];
    const int stride = KERNEL_PARAMS[5];
    const int padding = KERNEL_PARAMS[6];
    const int dilation = KERNEL_PARAMS[7];

    // Calculate batch and channel indices
    const int batch_idx = bz / ((channels + 7) / 8);
    const int channel_base = (bz % ((channels + 7) / 8)) * 8;

    // Calculate output coordinates
    const int out_x = bx * 32 + tx;
    const int out_y = by * 32 + ty;

    if (out_x >= output_width || out_y >= output_height || batch_idx >= batch_size) {
        return;
    }

    // Input strides
    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int input_stride_height = input_width;

    // Calculate input window start positions
    const int in_x_start = out_x * stride - padding;
    const int in_y_start = out_y * stride - padding;

    // Process up to 8 channels per thread
    #pragma unroll
    for (int c = 0; c < 8 && channel_base + c < channels; c++) {
        const int channel_idx = channel_base + c;
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        const int base_offset = batch_idx * input_stride_batch + channel_idx * input_stride_channel;

        // Compute max value for current window
        #pragma unroll 4
        for (int ky = 0; ky < kernel_size; ky++) {
            const int in_y = in_y_start + ky * dilation;
            if (in_y >= 0 && in_y < input_height) {
                #pragma unroll 4
                for (int kx = 0; kx < kernel_size; kx++) {
                    const int in_x = in_x_start + kx * dilation;
                    if (in_x >= 0 && in_x < input_width) {
                        const int input_idx = base_offset + in_y * input_stride_height + in_x;
                        max_val = max(max_val, __ldg(&input[input_idx]));
                    }
                }
            }
        }

        // Write output
        const int output_idx = batch_idx * channels * output_height * output_width +
                             channel_idx * output_height * output_width +
                             out_y * output_width +
                             out_x;
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

    // Copy parameters to constant memory
    int h_kernel_params[8] = {
        input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, dilation
    };
    cudaMemcpyToSymbol(KERNEL_PARAMS, h_kernel_params, sizeof(int) * 8);

    // Configure grid and block dimensions
    dim3 threads(32, 32);
    dim3 blocks(
        (output_width + 31) / 32,
        (output_height + 31) / 32,
        batch_size * ((channels + 7) / 8)
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}