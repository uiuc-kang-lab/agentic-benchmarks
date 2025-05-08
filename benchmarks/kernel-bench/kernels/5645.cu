#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void warp_optimized_maxpool2d_kernel(
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
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    for (int idx = gidx; idx < total; idx += grid_stride) {
        const int ow = idx % output_width;
        const int oh = (idx / output_width) % output_height;
        const int c = (idx / (output_width * output_height)) % channels;
        const int b = idx / (output_width * output_height * channels);

        const int ih_base = oh * stride - padding;
        const int iw_base = ow * stride - padding;
        
        // Precompute valid kernel ranges to avoid divergent branching
        const int kh_start = max(0, (-ih_base + dilation - 1) / dilation);
        const int kh_end = min(kernel_size, (input_height - ih_base + dilation - 1) / dilation);
        const int kw_start = max(0, (-iw_base + dilation - 1) / dilation);
        const int kw_end = min(kernel_size, (input_width - iw_base + dilation - 1) / dilation);

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const int plane_offset = b * channels * input_height * input_width 
                               + c * input_height * input_width;

        // Uniform loops with precomputed ranges
        for (int kh = kh_start; kh < kh_end; ++kh) {
            const int ih = ih_base + kh * dilation;
            const int row_offset = plane_offset + ih * input_width;
            
            for (int kw = kw_start; kw < kw_end; ++kw) {
                const int iw = iw_base + kw * dilation;
                const scalar_t val = input[row_offset + iw];
                max_val = max(max_val, val);
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor warp_optimized_maxpool2d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_optimized_maxpool2d", ([&] {
        warp_optimized_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_maxpool2d_forward, "Warp-optimized MaxPool2D forward (CUDA)");
}
