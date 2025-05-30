#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void coalesced_maxpool2d_kernel(
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

    const int bc = blockIdx.x;
    const int b = bc / channels;
    const int c = bc % channels;
    const int oh = blockIdx.y;

    if (b >= batch_size || c >= channels || oh >= output_height) return;
    
    const int input_plane_offset = b * channels * input_height * input_width + 
                                  c * input_height * input_width;
    const int output_plane_stride = output_height * output_width;

    for (int ow = threadIdx.x; ow < output_width; ow += blockDim.x) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = input_plane_offset + ih * input_width + iw;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        const int output_idx = bc * output_plane_stride + oh * output_width + ow;
        output[output_idx] = max_val;
    }
}

torch::Tensor coalesced_maxpool2d_cuda_forward(
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

    const int total_bc = batch_size * channels;
    const int threads = 256;
    dim3 grid(total_bc, output_height);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_maxpool2d_forward", ([&] {
        coalesced_maxpool2d_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &coalesced_maxpool2d_cuda_forward, "Coalesced Max Pool 2D forward (CUDA)");
}