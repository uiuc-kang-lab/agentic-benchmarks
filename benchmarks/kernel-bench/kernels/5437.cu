#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

const int NUM_STREAMS = 4;

template <typename scalar_t>
__global__ void pipelined_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
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
    const int dilation,
    const int stream_batch_size
) {
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = stream_batch_size * channels * output_height * output_width;
    if (output_idx >= total_elements) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding + kw * dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                max_val = max(max_val, __ldg(&input[input_idx]));
            }
        }
    }

    output[output_idx] = max_val;
}

torch::Tensor pipelined_max_pool2d_cuda_forward(
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

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int stream_batch_size = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pipelined_max_pool2d_cuda_forward", ([&] {
        for (int i = 0; i < NUM_STREAMS; i++) {
            const int stream_start = i * stream_batch_size;
            const int current_batch_size = std::min(stream_batch_size, batch_size - stream_start);
            
            if (current_batch_size <= 0) continue;

            const int elements_per_stream = current_batch_size * channels * output_height * output_width;
            const int blocks = (elements_per_stream + threads - 1) / threads;

            const scalar_t* input_ptr = input.data_ptr<scalar_t>() + stream_start * channels * input_height * input_width;
            scalar_t* output_ptr = output.data_ptr<scalar_t>() + stream_start * channels * output_height * output_width;

            pipelined_max_pool2d_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                input_ptr,
                output_ptr,
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
                current_batch_size
            );
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_max_pool2d_cuda_forward, "Pipelined Max Pool 2D forward (CUDA)");
}