#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
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

    const int BLOCK_SIZE = 16;
    const int shared_height = BLOCK_SIZE + kernel_size - 1;
    const int shared_width = BLOCK_SIZE + kernel_size - 1;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_x = blockIdx.x % ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int block_y = (blockIdx.x / ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE)) % ((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int c = (blockIdx.x / ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE * (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE)) % channels;
    const int b = blockIdx.x / (channels * ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE));

    const int base_h = block_y * BLOCK_SIZE * stride - padding;
    const int base_w = block_x * BLOCK_SIZE * stride - padding;

    // Load data into shared memory
    for (int i = tid; i < shared_height * shared_width; i += blockDim.x * blockDim.y) {
        const int sh = i / shared_width;
        const int sw = i % shared_width;
        const int ih = base_h + sh;
        const int iw = base_w + sw;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_input[sh * shared_width + sw] = input[b * (channels * input_height * input_width) +
                                                       c * (input_height * input_width) +
                                                       ih * input_width + iw];
        } else {
            shared_input[sh * shared_width + sw] = -std::numeric_limits<scalar_t>::infinity();
        }
    }

    __syncthreads();

    // Compute max pooling
    const int out_h = block_y * BLOCK_SIZE + threadIdx.y;
    const int out_w = block_x * BLOCK_SIZE + threadIdx.x;

    if (out_h < output_height && out_w < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        const int shared_h_offset = threadIdx.y * stride;
        const int shared_w_offset = threadIdx.x * stride;

        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int shared_idx = (shared_h_offset + kh) * shared_width + shared_w_offset + kw;
                max_val = max(max_val, shared_input[shared_idx]);
            }
        }

        const int output_idx = b * (channels * output_height * output_width) +
                              c * (output_height * output_width) +
                              out_h * output_width + out_w;
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
    const int blocks_x = (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_y = (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks = batch_size * channels * blocks_x * blocks_y;

    const int shared_mem_size = (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1) * sizeof(float);

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