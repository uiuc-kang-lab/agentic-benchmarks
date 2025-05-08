#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void warp_aligned_maxpool2d_kernel(
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
    // Use 32x4 thread block for warp alignment
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * 32 + tx;
    const int oh = blockIdx.y * 4 + ty;
    const int bc = blockIdx.z;

    if (oh >= output_height) return;  // Uniform warp exit

    const int b = bc / channels;
    const int c = bc % channels;
    
    const int input_batch_offset = b * channels * input_height * input_width;
    const int input_channel_offset = c * input_height * input_width;
    const int base_input_offset = input_batch_offset + input_channel_offset;

    // Pre-compute boundary conditions for the entire warp
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    
    // Compute valid ranges once per thread
    const int ih_end = min(ih_start + (KERNEL_SIZE-1) * dilation + 1, input_height);
    const int iw_end = min(iw_start + (KERNEL_SIZE-1) * dilation + 1, input_width);
    const int ih_valid_start = max(0, ih_start);
    const int iw_valid_start = max(0, iw_start);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if (ow < output_width) {  // Uniform condition per warp
        if constexpr (KERNEL_SIZE == 2) {
            // Vectorized handling for 2x2 kernel
            const bool valid_h = (ih_valid_start < ih_end);
            const bool valid_w = (iw_valid_start < iw_end);
            
            if (valid_h && valid_w) {
                const int row1 = base_input_offset + ih_valid_start * input_width;
                const int row2 = base_input_offset + min(ih_valid_start + dilation, input_height-1) * input_width;
                
                scalar_t val1 = __ldg(&input[row1 + iw_valid_start]);
                scalar_t val2 = __ldg(&input[row1 + min(iw_valid_start + dilation, input_width-1)]);
                scalar_t val3 = __ldg(&input[row2 + iw_valid_start]);
                scalar_t val4 = __ldg(&input[row2 + min(iw_valid_start + dilation, input_width-1)]);
                
                max_val = max(max(max(val1, val2), val3), val4);
            }
        } else {
            // Uniform loop bounds for larger kernels
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                const int ih = ih_start + kh * dilation;
                if (ih >= ih_valid_start && ih < ih_end) {
                    const int row_offset = base_input_offset + ih * input_width;
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        const int iw = iw_start + kw * dilation;
                        if (iw >= iw_valid_start && iw < iw_end) {
                            max_val = max(max_val, __ldg(&input[row_offset + iw]));
                        }
                    }
                }
            }
        }
    }

    if (ow < output_width) {  // Uniform store condition
        output[bc * output_height * output_width + oh * output_width + ow] = max_val;
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

    // Use 32x4 threads for warp alignment
    dim3 threads(32, 4);
    dim3 blocks(
        (output_width + 31) / 32,
        (output_height + 3) / 4,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            warp_aligned_maxpool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            warp_aligned_maxpool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            warp_aligned_maxpool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Warp-aligned Max Pool 2D forward (CUDA)");
}