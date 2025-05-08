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
    const int start_bc,
    const int end_bc
) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = start_bc + blockIdx.z;

    if (bc >= end_bc || oh >= output_height || ow >= output_width) return;

    const int b = bc / channels;
    const int c = bc % channels;
    
    if (b >= batch_size || c >= channels) return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;

    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            const int ih = base_ih + kh * dilation;
            const bool valid_h = ih >= 0 && ih < input_height;
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                const int iw = base_iw + kw * dilation;
                if (valid_h && iw >= 0 && iw < input_width)
                    max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    } else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int ih = base_ih + kh * dilation;
            const bool valid_h = ih >= 0 && ih < input_height;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                const int iw = base_iw + kw * dilation;
                if (valid_h && iw >= 0 && iw < input_width)
                    max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    } else {
        #pragma unroll 4
        for (int kh = 0; kh < kernel_size; kh++) {
            const int ih = base_ih + kh * dilation;
            const bool valid_h = ih >= 0 && ih < input_height;
            #pragma unroll 4
            for (int kw = 0; kw < kernel_size; kw++) {
                const int iw = base_iw + kw * dilation;
                if (valid_h && iw >= 0 && iw < input_width)
                    max_val = max(max_val, input_channel[ih * input_width + iw]);
            }
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    const dim3 threads(32, 8);
    const int total_bc = batch_size * channels;
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int bc_per_stream = (total_bc + num_streams - 1) / num_streams;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        for (int i = 0; i < num_streams; ++i) {
            const int start = i * bc_per_stream;
            const int end = std::min(start + bc_per_stream, total_bc);
            if (start >= end) continue;

            const dim3 blocks(
                (output_width + threads.x - 1) / threads.x,
                (output_height + threads.y - 1) / threads.y,
                end - start
            );

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
                start,
                end
            );
        }
    }));

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with stream optimization");
}
