#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int BLOCK_SIZE = 256>
__global__ void max_pool2d_streamed_kernel(
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
    const int chunk_start,
    const int chunk_size
) {
    __shared__ scalar_t shared_input[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = chunk_size * output_height * output_width;
    
    if (global_idx >= total_elements) return;

    const int c = chunk_start + (global_idx / (output_height * output_width));
    const int oh = (global_idx / output_width) % output_height;
    const int ow = global_idx % output_width;
    const int b = blockIdx.y;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int ih = oh * stride - padding + kh * dilation;
            const int iw = ow * stride - padding + kw * dilation;

            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                shared_input[tid] = input[input_idx];
                __syncthreads();
                
                max_val = max(max_val, shared_input[tid]);
                __syncthreads();
            }
        }
    }

    if (c < channels) {
        const int output_idx = b * (channels * output_height * output_width) +
                             c * (output_height * output_width) +
                             oh * output_width +
                             ow;
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

    const int num_streams = 4;
    const int chunk_size = (channels + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        for (int i = 0; i < num_streams; i++) {
            const int chunk_start = i * chunk_size;
            const int current_chunk_size = std::min(chunk_size, channels - chunk_start);
            
            if (current_chunk_size <= 0) continue;
            
            const int elements_per_chunk = current_chunk_size * output_height * output_width;
            const int blocks_x = (elements_per_chunk + threads - 1) / threads;
            const dim3 blocks(blocks_x, batch_size);
            
            max_pool2d_streamed_kernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
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
                chunk_start,
                current_chunk_size
            );
        }
    }));

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}