#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Device function to compute the output indices (b, c, oh, ow) from a flat index
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

// Device function to perform max pooling over the kernel window for one output element
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_window_max(
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

    const int input_plane = input_height * input_width;
    const int offset = b * channels * input_plane + c * input_plane;
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                int idx = offset + ih * input_width + iw;
                scalar_t val = input[idx];
                max_val = (val > max_val) ? val : max_val;
            }
        }
    }
    return max_val;
}

// CUDA kernel using grid-stride loop with modular device functions
template <typename scalar_t>
__global__ void modularized_maxpool2d_kernel(
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

    const int total_elements = batch_size * channels * output_height * output_width;
    const int grid_stride = blockDim.x * gridDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_elements; index += grid_stride) {
        int b, c, oh, ow;
        get_output_indices<scalar_t>(index, output_width, output_height, channels, b, c, oh, ow);
        output[index] = compute_window_max(input, b, c, oh, ow,
                                             input_height, input_width,
                                             kernel_size, stride, padding, dilation, channels);
    }
}

// Host function to launch the kernel
torch::Tensor modularized_maxpool2d_cuda_forward(
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "modularized_maxpool2d_cuda_forward", ([&] {
        modularized_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modularized_maxpool2d_cuda_forward, "Modularized MaxPool2D forward (CUDA)");
}
