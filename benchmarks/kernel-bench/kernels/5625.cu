#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void coalesced_max_pool2d_kernel(
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
    __shared__ scalar_t shared_max[32][8];
    
    const int tx = threadIdx.x;  // 32 threads for coalesced access
    const int ty = threadIdx.y;  // 8 threads
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int bc = blockIdx.z;
    
    if (ow >= output_width || oh >= output_height) return;

    const int b = bc / channels;
    const int c = bc % channels;
    
    const int batch_offset = b * channels * input_height * input_width;
    const int channel_offset = c * input_height * input_width;
    const int input_base = batch_offset + channel_offset;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    if constexpr (KERNEL_SIZE == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_offset = ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 2; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_base + row_offset + iw]));
                    }
                }
            }
        }
    } else if constexpr (KERNEL_SIZE == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_offset = ih * input_width;
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_base + row_offset + iw]));
                    }
                }
            }
        }
    } else {
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            const int ih = ih_start + kh * dilation;
            if (ih >= 0 && ih < input_height) {
                const int row_offset = ih * input_width;
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    const int iw = iw_start + kw * dilation;
                    if (iw >= 0 && iw < input_width) {
                        max_val = max(max_val, __ldg(&input[input_base + row_offset + iw]));
                    }
                }
            }
        }
    }

    shared_max[tx][ty] = max_val;
    __syncthreads();

    if (ow < output_width && oh < output_height) {
        const int output_idx = bc * output_height * output_width + oh * output_width + ow;
        output[output_idx] = shared_max[tx][ty];
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

    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        if (kernel_size == 2) {
            coalesced_max_pool2d_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else if (kernel_size == 3) {
            coalesced_max_pool2d_kernel<scalar_t, 3><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, channels,
                input_height, input_width,
                output_height, output_width,
                stride, padding, dilation
            );
        } else {
            coalesced_max_pool2d_kernel<scalar_t, -1><<<blocks, threads>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with coalesced memory access (CUDA)");
}