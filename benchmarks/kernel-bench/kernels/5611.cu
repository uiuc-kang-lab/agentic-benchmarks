#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void max_pool2d_streamed_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_start,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;

    if (ow >= output_width || oh >= output_height) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int h_start = oh * stride - padding;
    const int w_start = ow * stride - padding;

    for (int b = batch_start; b < batch_start + batch_size; ++b) {
        const int input_base = b * channels * input_height * input_width
                            + c * input_height * input_width;

        if constexpr (KERNEL_SIZE == 2) {
            #pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                const int ih = h_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int row = input_base + ih * input_width;
                    #pragma unroll
                    for (int kw = 0; kw < 2; ++kw) {
                        const int iw = w_start + kw * dilation;
                        if (iw >= 0 && iw < input_width)
                            max_val = fmaxf(max_val, __ldg(input + row + iw));
                    }
                }
            }
        } else if constexpr (KERNEL_SIZE == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = h_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int row = input_base + ih * input_width;
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        const int iw = w_start + kw * dilation;
                        if (iw >= 0 && iw < input_width)
                            max_val = fmaxf(max_val, __ldg(input + row + iw));
                    }
                }
            }
        } else {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                const int ih = h_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int row = input_base + ih * input_width;
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                        const int iw = w_start + kw * dilation;
                        if (iw >= 0 && iw < input_width)
                            max_val = fmaxf(max_val, __ldg(input + row + iw));
                    }
                }
            }
        }
    }

    const int out_idx = (batch_start * channels + c) * output_height * output_width
                      + oh * output_width + ow;
    output[out_idx] = max_val;
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

    const auto output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1;
    const auto output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    const int num_streams = 4;
    const int batches_per_stream = (batch_size + num_streams - 1) / num_streams;
    std::vector<torch::cuda::CUDAStream> streams;
    for (int i = 0; i < num_streams; ++i) {
        streams.push_back(torch::cuda::getStreamFromPool());
    }

    const dim3 threads(32, 8);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool_forward", ([&] {
        for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
            const int batch_start = stream_idx * batches_per_stream;
            const int current_batch_size = std::min(batches_per_stream, batch_size - batch_start);
            if (current_batch_size <= 0) break;

            const dim3 blocks(
                (output_width + threads.x - 1) / threads.x,
                (output_height + threads.y - 1) / threads.y,
                channels
            );

            auto stream = streams[stream_idx];
            if (kernel_size == 2) {
                max_pool2d_streamed_kernel<scalar_t, 2>
                    <<<blocks, threads, 0, stream.stream()>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_start,
                        current_batch_size,
                        channels,
                        input_height,
                        input_width,
                        output_height,
                        output_width,
                        stride,
                        padding,
                        dilation
                    );
            } else if (kernel_size == 3) {
                max_pool2d_streamed_kernel<scalar_t, 3>
                    <<<blocks, threads, 0, stream.stream()>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_start,
                        current_batch_size,
                        channels,
                        input_height,
                        input_width,
                        output_height,
                        output_width,
                        stride,
                        padding,
                        dilation
                    );
            } else {
                max_pool2d_streamed_kernel<scalar_t, -1>
                    <<<blocks, threads, 0, stream.stream()>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_start,
                        current_batch_size,
                        channels,
                        input_height,
                        input_width,
                        output_height,
                        output_width,
                        stride,
                        padding,
                        dilation
                    );
            }
        }
    }));

    for (auto& stream : streams) {
        stream.synchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward with streamed execution (CUDA)");
}
