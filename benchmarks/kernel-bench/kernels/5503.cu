#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ void load_input_tile(
    const scalar_t* input,
    scalar_t* shared_data,
    const int tid,
    const int block_size,
    const int tile_size,
    const int input_height,
    const int input_width,
    const int ih_start,
    const int iw_start,
    const int channel_offset
) {
    for (int i = tid; i < tile_size * tile_size; i += block_size) {
        const int tile_h = i / tile_size;
        const int tile_w = i % tile_size;
        const int ih = ih_start + tile_h;
        const int iw = iw_start + tile_w;
        
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            shared_data[tile_h * tile_size + tile_w] = input[channel_offset + ih * input_width + iw];
        } else {
            shared_data[tile_h * tile_size + tile_w] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_max_from_tile(
    const scalar_t* shared_data,
    const int kernel_size,
    const int tile_size,
    const int local_h,
    const int local_w
) {
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int tile_idx = (local_h + kh) * tile_size + (local_w + kw);
            max_val = max(max_val, shared_data[tile_idx]);
        }
    }
    
    return max_val;
}

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
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int tile_size = 16;  // Adjust based on kernel_size and stride
    
    const int output_idx = blockIdx.x * block_size + tid;
    if (output_idx >= batch_size * channels * output_height * output_width) return;
    
    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);
    
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    const int channel_offset = (b * channels + c) * input_height * input_width;
    
    load_input_tile<scalar_t>(
        input, shared_data, tid, block_size, tile_size,
        input_height, input_width, ih_start, iw_start, channel_offset
    );
    
    __syncthreads();
    
    const int local_h = (oh * stride - padding) - ih_start;
    const int local_w = (ow * stride - padding) - iw_start;
    
    output[output_idx] = compute_max_from_tile<scalar_t>(
        shared_data, kernel_size, tile_size, local_h, local_w
    );
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

    const int threads = 256;
    const int blocks = (batch_size * channels * output_height * output_width + threads - 1) / threads;
    const int shared_memory_size = 16 * 16 * sizeof(float);  // tile_size * tile_size

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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