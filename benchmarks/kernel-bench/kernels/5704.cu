#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel using shared memory with synchronization optimization

template<typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_shared_memory_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ scalar_t shared_mem[];

    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    const int input_offset = (b * channels + c) * input_height * input_width;
    const int base_h = oh * stride - padding;
    const int base_w = ow * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
        const int ih = base_h + kh * dilation;
        if (ih >= 0 && ih < input_height) {
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int iw = base_w + kw * dilation;
                if (iw >= 0 && iw < input_width) {
                    int idx = input_offset + ih * input_width + iw;
                    max_val = fmaxf(max_val, __ldg(&input[idx]));
                }
            }
        }
    }

    shared_mem[threadIdx.y * blockDim.x + threadIdx.x] = max_val;
    __syncthreads();  // Ensure all threads have written their results to shared memory

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        scalar_t block_max = -std::numeric_limits<scalar_t>::infinity();
        for (int i = 0; i < blockDim.x * blockDim.y; ++i) {
            block_max = fmaxf(block_max, shared_mem[i]);
        }
        output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = block_max;
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

    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(32, 8);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    size_t shared_mem_size = block.x * block.y * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_shared_memory_kernel<scalar_t, 2><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
                break;
            case 3:
                max_pool2d_shared_memory_kernel<scalar_t, 3><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
                break;
            default:
                max_pool2d_shared_memory_kernel<scalar_t, -1><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    batch_size, channels, input_height, input_width,
                    output_height, output_width, stride, padding, dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D with shared memory optimization (CUDA)");
}