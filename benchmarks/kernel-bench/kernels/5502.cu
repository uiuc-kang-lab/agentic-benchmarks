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
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int output_idx = blockIdx.x * block_size + tid;

    // Calculate output indices
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    if (output_idx >= batch_size * channels * output_height * output_width) return;

    // Calculate the input region that needs to be loaded into shared memory
    const int start_h = oh * stride - padding;
    const int start_w = ow * stride - padding;
    const int end_h = start_h + (kernel_size - 1) * dilation + 1;
    const int end_w = start_w + (kernel_size - 1) * dilation + 1;

    const int tile_width = blockDim.x + (kernel_size - 1) * dilation;
    const int tile_height = blockDim.x + (kernel_size - 1) * dilation;

    // Load input data into shared memory
    const int input_offset = b * channels * input_height * input_width +
                           c * input_height * input_width;

    for (int h = start_h + tid; h < end_h; h += block_size) {
        for (int w = start_w; w < end_w; w++) {
            if (h >= 0 && h < input_height && w >= 0 && w < input_width) {
                const int shared_idx = (h - start_h) * tile_width + (w - start_w);
                const int input_idx = input_offset + h * input_width + w;
                shared_input[shared_idx] = input[input_idx];
            }
            else {
                const int shared_idx = (h - start_h) * tile_width + (w - start_w);
                shared_input[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    __syncthreads();

    // Compute max pooling using shared memory
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int h = kh * dilation;
            const int w = kw * dilation;
            const int shared_idx = h * tile_width + w;
            max_val = max(max_val, shared_input[shared_idx]);
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;

    // Calculate shared memory size
    const int tile_width = threads + (kernel_size - 1) * dilation;
    const int tile_height = threads + (kernel_size - 1) * dilation;
    const int shared_memory_size = tile_width * tile_height * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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