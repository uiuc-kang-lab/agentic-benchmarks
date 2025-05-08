#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel using shared memory to reduce global memory accesses
// and synchronize threads only when necessary

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_shared_memory_kernel(
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
    extern __shared__ scalar_t shared_input[];

    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width) return;

    const int input_base = b * channels * input_height * input_width + 
                          c * input_height * input_width;

    const int shared_width = blockDim.x + KERNEL_SIZE - 1;
    const int shared_height = blockDim.y + KERNEL_SIZE - 1;

    // Load input into shared memory
    for (int kh = threadIdx.y; kh < shared_height; kh += blockDim.y) {
        for (int kw = threadIdx.x; kw < shared_width; kw += blockDim.x) {
            int ih = oh * stride - padding + kh * dilation - threadIdx.y * dilation;
            int iw = ow * stride - padding + kw * dilation - threadIdx.x * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                shared_input[kh * shared_width + kw] = input[input_base + ih * input_width + iw];
            } else {
                shared_input[kh * shared_width + kw] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    __syncthreads();

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
            max_val = fmaxf(max_val, shared_input[(threadIdx.y + kh) * shared_width + (threadIdx.x + kw)]);
        }
    }

    output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const dim3 block(32, 8);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    const int shared_mem_size = (block.x + kernel_size - 1) * (block.y + kernel_size - 1) * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        switch(kernel_size) {
            case 2:
                max_pool2d_shared_memory_kernel<scalar_t, 2><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
                break;
            case 3:
                max_pool2d_shared_memory_kernel<scalar_t, 3><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
                break;
            default:
                max_pool2d_shared_memory_kernel<scalar_t, -1><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    stride,
                    padding,
                    dilation
                );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Shared Memory Optimized Max Pool 2D forward (CUDA)");
}
