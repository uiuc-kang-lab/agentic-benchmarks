#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Using shared memory to reduce global memory access
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
    extern __shared__ scalar_t shared_input[];

    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * blockDim.x + tid;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    const int shared_height = kernel_size + (blockDim.x - 1) / output_width;
    const int shared_width = kernel_size;

    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;

    // Load data into shared memory
    for (int kh = 0; kh < shared_height; ++kh) {
        for (int kw = 0; kw < shared_width; ++kw) {
            const int ih = ih_base + kh * dilation;
            const int iw = iw_base + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                      c * (input_height * input_width) +
                                      ih * input_width +
                                      iw;
                shared_input[tid * kernel_size * kernel_size + kh * kernel_size + kw] = input[input_idx];
            } else {
                shared_input[tid * kernel_size * kernel_size + kh * kernel_size + kw] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    // Compute max pooling
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            max_val = max(max_val, shared_input[tid * kernel_size * kernel_size + kh * kernel_size + kw]);
        }
    }

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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    const int shared_mem_size = threads * kernel_size * kernel_size * sizeof(float);

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
