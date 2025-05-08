#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function to compute the (b, c, oh, ow) indices for the output tensor
template <typename scalar_t>
__device__ __forceinline__ void get_output_indices(const int index,
                                                     const int output_width,
                                                     const int output_height,
                                                     const int channels,
                                                     int &b, int &c, int &oh, int &ow) {
    ow = index % output_width;
    oh = (index / output_width) % output_height;
    c  = (index / (output_width * output_height)) % channels;
    b  = index / (output_width * output_height * channels);
}

// Device function to compute the max value in a pooling window for one output element
// It uses precomputed pooling start indices (start_h, start_w) and iterates through the kernel window.
template <typename scalar_t>
__device__ __forceinline__ scalar_t max_in_pooling_window(
    const scalar_t* __restrict__ input,
    const int b,
    const int c,
    const int oh,
    const int ow,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int channels) {

    const int channel_area = input_height * input_width;
    const int offset = b * channels * channel_area + c * channel_area;
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Compute the starting indices for the pooling window
    const int start_h = oh * stride - padding;
    const int start_w = ow * stride - padding;

    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = start_h + kh * dilation;
        if (ih < 0 || ih >= input_height)
            continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = start_w + kw * dilation;
            if (iw < 0 || iw >= input_width)
                continue;
            int idx = offset + ih * input_width + iw;
            scalar_t val = input[idx];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    return max_val;
}

// Modular kernel that uses the above device functions in a grid-stride loop
template <typename scalar_t>
__global__ void modular_maxpool2d_kernel(
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
    const int dilation) {

    const int total = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridStride = blockDim.x * gridDim.x;

    for (; idx < total; idx += gridStride) {
        int b, c, oh, ow;
        get_output_indices<scalar_t>(idx, output_width, output_height, channels, b, c, oh, ow);
        output[idx] = max_in_pooling_window<scalar_t>(
            input, b, c, oh, ow,
            input_height, input_width,
            kernel_size, stride, padding, dilation,
            channels);
    }
}

// Host function to launch the modularized max pooling kernel
torch::Tensor modular_maxpool2d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modular_maxpool2d_forward", ([&] {
        modular_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
            dilation);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_maxpool2d_forward, "Modular MaxPool2D forward (CUDA)");
}
