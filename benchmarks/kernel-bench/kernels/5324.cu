#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
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
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    // Calculate input region boundaries
    const int start_h = oh * stride - padding;
    const int start_w = ow * stride - padding;
    const int end_h = min(start_h + kernel_size * dilation, input_height);
    const int end_w = min(start_w + kernel_size * dilation, input_width);

    // Calculate shared memory dimensions for this block
    const int BLOCK_H = blockDim.y;
    const int BLOCK_W = blockDim.x;
    const int shared_h = ((end_h - start_h + BLOCK_H - 1) / BLOCK_H) * BLOCK_H;
    const int shared_w = ((end_w - start_w + BLOCK_W - 1) / BLOCK_W) * BLOCK_W;

    // Load input data into shared memory
    const int input_batch_offset = b * (channels * input_height * input_width);
    const int input_channel_offset = c * (input_height * input_width);

    for (int h = threadIdx.y; h < shared_h; h += BLOCK_H) {
        for (int w = threadIdx.x; w < shared_w; w += BLOCK_W) {
            const int ih = start_h + h;
            const int iw = start_w + w;
            const int shared_idx = h * shared_w + w;
            
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = input_batch_offset + input_channel_offset + ih * input_width + iw;
                shared_input[shared_idx] = input[input_idx];
            } else {
                shared_input[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    __syncthreads();

    // Compute max pooling using shared memory
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        const int ih = kh * dilation;
        if (start_h + ih >= 0 && start_h + ih < input_height) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = kw * dilation;
                if (start_w + iw >= 0 && start_w + iw < input_width) {
                    const int shared_idx = ih * shared_w + iw;
                    max_val = max(max_val, shared_input[shared_idx]);
                }
            }
        }
    }

    if (output_idx < batch_size * channels * output_height * output_width) {
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

    const int BLOCK_SIZE = 16;
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks(
        (batch_size * channels * output_height * output_width + threads.x - 1) / threads.x,
        1
    );

    const int shared_memory_size = (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with shared memory (CUDA)");
}