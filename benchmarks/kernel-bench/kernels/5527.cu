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

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int block_width = blockDim.x;
    const int block_height = blockDim.y;

    const int ow_start = bx * block_width + tx;
    const int oh_start = by * block_height + ty;
    const int c = bz % channels;
    const int b = bz / channels;

    if (ow_start >= output_width || oh_start >= output_height) return;

    const int ih_start = oh_start * stride - padding;
    const int iw_start = ow_start * stride - padding;
    const int ih_end = ih_start + kernel_size * dilation;
    const int iw_end = iw_start + kernel_size * dilation;

    const int shared_width = block_width * stride + (kernel_size - 1) * dilation + 1;
    const int shared_height = block_height * stride + (kernel_size - 1) * dilation + 1;

    for (int ih = ih_start + ty; ih <= ih_end; ih += block_height) {
        for (int iw = iw_start + tx; iw <= iw_end; iw += block_width) {
            const int shared_idx = (ih - ih_start) * shared_width + (iw - iw_start);
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width + iw;
                shared_input[shared_idx] = input[input_idx];
            } else {
                shared_input[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }

    __syncthreads();

    if (ow_start < output_width && oh_start < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = kw * dilation;
                const int shared_idx = ih * shared_width + iw;
                max_val = max(max_val, shared_input[shared_idx]);
            }
        }

        const int output_idx = b * (channels * output_height * output_width) +
                             c * (output_height * output_width) +
                             oh_start * output_width +
                             ow_start;
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

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    const int shared_mem_size = 
        (threads.x * stride + (kernel_size - 1) * dilation + 1) *
        (threads.y * stride + (kernel_size - 1) * dilation + 1) *
        sizeof(float);

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