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
    extern __shared__ scalar_t shared_mem[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 8;
    
    const int oh = by * TILE_HEIGHT + ty;
    const int ow = bx * TILE_WIDTH + tx;
    const int c = bz % channels;
    const int b = bz / channels;

    if (oh >= output_height || ow >= output_width) return;

    const int tile_h_start = oh * stride - padding;
    const int tile_w_start = ow * stride - padding;
    const int tile_size = (kernel_size + stride - 1);

    const int input_stride_batch = channels * input_height * input_width;
    const int input_stride_channel = input_height * input_width;
    const int base_offset = b * input_stride_batch + c * input_stride_channel;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int shared_idx = ty * (TILE_WIDTH + kernel_size - 1) + tx;
    const int shared_stride = TILE_HEIGHT * (TILE_WIDTH + kernel_size - 1);

    if (shared_idx < tile_size * tile_size) {
        const int rel_h = shared_idx / tile_size;
        const int rel_w = shared_idx % tile_size;
        const int ih = tile_h_start + rel_h;
        const int iw = tile_w_start + rel_w;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_mem[shared_idx] = __ldg(&input[base_offset + ih * input_width + iw]);
        } else {
            shared_mem[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
        }
    }

    if (kernel_size > 1) {
        __syncthreads();
    }

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = oh * stride - padding + kh * dilation;
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int iw = ow * stride - padding + kw * dilation;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int shared_h = kh;
                const int shared_w = tx + kw;
                
                if (shared_h < TILE_HEIGHT && shared_w < TILE_WIDTH + kernel_size - 1) {
                    max_val = max(max_val, shared_mem[shared_h * (TILE_WIDTH + kernel_size - 1) + shared_w]);
                } else {
                    max_val = max(max_val, __ldg(&input[base_offset + ih * input_width + iw]));
                }
            }
        }
    }

    if (oh < output_height && ow < output_width) {
        const int output_idx = b * channels * output_height * output_width +
                             c * output_height * output_width +
                             oh * output_width +
                             ow;
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

    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int shared_mem_size = threads.y * (threads.x + kernel_size - 1) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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