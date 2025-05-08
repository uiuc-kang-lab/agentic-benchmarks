#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_streams_kernel(
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
    const int dilation,
    const int start_idx,
    const int elements_per_stream
) {
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_idx = start_idx + local_idx;
    if (output_idx >= start_idx + elements_per_stream) return;
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding + kw * dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward_streams(
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

    const int output_height = static_cast<int>(((static_cast<int64_t>(input_height) + 2LL * padding - dilation * (kernel_size - 1) - 1) / stride) + 1);
    const int output_width = static_cast<int>(((static_cast<int64_t>(input_width) + 2LL * padding - dilation * (kernel_size - 1) - 1) / stride) + 1);

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int elements_per_stream = (total_elements + num_streams - 1) / num_streams;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_streams_forward", ([&] {
        for (int i = 0; i < num_streams; ++i) {
            const int start_idx = i * elements_per_stream;
            const int elements = std::min(elements_per_stream, total_elements - start_idx);
            const int blocks = (elements + threads - 1) / threads;

            if (blocks > 0) {
                max_pool2d_streams_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
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
                    dilation,
                    start_idx,
                    elements
                );
            }
        }
    }));

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward_streams, "Max Pool 2D forward with streams (CUDA)");
}