#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[8];

// Helper function to compute a flattened index for the input tensor
__device__ int input_index(int b, int c, int h, int w, int channels, int height, int width) {
    return ((b * channels + c) * height + h) * width + w;
}

template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    extern __shared__ scalar_t tile[]; // Dynamically allocated shared memory

    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) return;

    const int ow = idx % output_width;
    const int oh = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    const int shared_input_height = kernel_size * stride;
    const int shared_input_width = kernel_size * stride;
    const int shared_input_size = shared_input_height * shared_input_width;

    // Load input data into shared memory
    for (int i = threadIdx.x; i < shared_input_size; i += blockDim.x) {
        int sh = (i / shared_input_width) + oh * stride - padding;
        int sw = (i % shared_input_width) + ow * stride - padding;
        if (sh >= 0 && sh < input_height && sw >= 0 && sw < input_width) {
            tile[i] = input[input_index(b, c, sh, sw, channels, input_height, input_width)];
        } else {
            tile[i] = -std::numeric_limits<scalar_t>::infinity(); // Padding with minimum value
        }
    }
    __syncthreads();

    // Perform the max pooling operation using shared memory
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int sh = kh * dilation;
            int sw = kw * dilation;
            max_val = max(max_val, tile[sh * shared_input_width + sw]);
        }
    }

    output[idx] = max_val;
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

    const int params[8] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(int) * 8);

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    const int shared_memory_size = kernel_size * stride * kernel_size * stride * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}
