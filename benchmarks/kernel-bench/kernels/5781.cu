#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int NUM_STREAMS = 4;
constexpr int CHUNK_SIZE = 32;

template <typename scalar_t>
__global__ void max_pool2d_stream_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int start_idx,
    const int elements_this_chunk,
    const int batch_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements_this_chunk) return;
    
    const int idx = start_idx + tid;
    const int bc = idx / (output_height * output_width);
    const int oh = (idx / output_width) % output_height;
    const int ow = idx % output_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 2; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = (bc * input_height + ih) * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    } else {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;
                
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = (bc * input_height + ih) * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor max_pool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    const int batch_channels = batch_size * channels;
    const int total_elements = batch_channels * output_height * output_width;
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_stream_forward", ([&] {
        for (int chunk_start = 0; chunk_start < total_elements; chunk_start += CHUNK_SIZE * NUM_STREAMS) {
            for (int s = 0; s < NUM_STREAMS; ++s) {
                const int start_idx = chunk_start + s * CHUNK_SIZE;
                if (start_idx >= total_elements) break;
                
                const int elements_this_chunk = min(CHUNK_SIZE, total_elements - start_idx);
                const int blocks = (elements_this_chunk + threads - 1) / threads;
                
                max_pool2d_stream_kernel<scalar_t><<<blocks, threads, 0, streams[s]>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    start_idx,
                    elements_this_chunk,
                    batch_channels,
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
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D stream pipelined forward (CUDA)");
}