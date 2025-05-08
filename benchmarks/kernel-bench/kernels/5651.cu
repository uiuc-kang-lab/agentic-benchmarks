#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void streamed_maxpool2d_kernel(
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
    
    const int total_elements = channels * output_height * output_width;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Each block handles one batch element
    if (bid >= batch_size) return;
    
    for (int out_idx = tid; out_idx < total_elements; out_idx += blockDim.x) {
        const int c = out_idx / (output_height * output_width);
        const int oh = (out_idx % (output_height * output_width)) / output_width;
        const int ow = out_idx % output_width;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const int input_offset = (bid * channels + c) * input_height * input_width;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const scalar_t val = input[input_offset + ih * input_width + iw];
                    max_val = (val > max_val) ? val : max_val;
                }
            }
        }
        output[(bid * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
    }
}

torch::Tensor streamed_maxpool2d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int num_streams) {
    
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Split batch elements across streams
    const int batch_per_stream = (batch_size + num_streams - 1) / num_streams;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "streamed_maxpool2d_forward", ([&] {
        for (int s = 0; s < num_streams; ++s) {
            const int batch_start = s * batch_per_stream;
            const int batch_end = std::min((s + 1) * batch_per_stream, batch_size);
            
            if (batch_start >= batch_end) continue;
            
            // Process this batch range in a separate stream
            streamed_maxpool2d_kernel<scalar_t>
                <<<batch_end - batch_start, threads, 0, streams[s]>>>(
                    input.data_ptr<scalar_t>() + batch_start * channels * input_height * input_width,
                    output.data_ptr<scalar_t>() + batch_start * channels * output_height * output_width,
                    batch_end - batch_start,
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
        }
    }));

    // Synchronize all streams
    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_maxpool2d_forward, "Streamed MaxPool 2D forward (CUDA)");
}
