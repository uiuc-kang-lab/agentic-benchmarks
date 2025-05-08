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
    // Calculate position using block and thread indices to ensure coalesced memory access
    const int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int c_idx = blockIdx.z % channels;
    const int b_idx = blockIdx.z / channels;

    // Early exit if outside bounds
    if (w_idx >= output_width || h_idx >= output_height || 
        b_idx >= batch_size || c_idx >= channels) return;

    // Calculate base offset for this batch and channel
    const int batch_offset = b_idx * channels * input_height * input_width;
    const int channel_offset = c_idx * input_height * input_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = h_idx * stride - padding + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            const int ih_offset = ih * input_width;
            
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = w_idx * stride - padding + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = batch_offset + channel_offset + ih_offset + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    // Write output using coalesced access pattern
    const int output_idx = b_idx * (channels * output_height * output_width) +
                          c_idx * (output_height * output_width) +
                          h_idx * output_width +
                          w_idx;
    output[output_idx] = max_val;
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

    // Configure thread blocks for coalesced memory access
    const dim3 threads(32, 8, 1);  // 32 threads along width dimension for coalescing
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
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