#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void tuned_grid_maxpool2d_kernel(
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

    // 2D grid organization for better spatial locality
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ow >= output_width || oh >= output_height || b >= batch_size) return;

    const int input_plane_offset = (b * channels + c) * input_height * input_width;
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll 4
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int ih = h_start + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        
        #pragma unroll 4
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int iw = w_start + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            
            const scalar_t val = input[input_plane_offset + ih * input_width + iw];
            max_val = max(max_val, val);
        }
    }

    output[((b * channels + c) * output_height + oh) * output_width + ow] = max_val;
}

torch::Tensor tuned_grid_maxpool2d_forward(
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
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // 2D thread blocks (32x8), grid covers spatial dimensions and batch+channels
    dim3 threads(32, 8);
    dim3 grids(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tuned_grid_maxpool2d_forward", ([&] {
        tuned_grid_maxpool2d_kernel<scalar_t><<<grids, threads>>>(
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
    m.def("forward", &tuned_grid_maxpool2d_forward, "Tuned Grid Max Pool 2D forward (CUDA)");
}
