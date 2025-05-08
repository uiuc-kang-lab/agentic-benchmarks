#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_shared_kernel(
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
    const int BLOCK_W = 32;
    const int BLOCK_H = 8;
    const int TILE_W = BLOCK_W + (kernel_size - 1) * dilation;
    const int TILE_H = BLOCK_H + (kernel_size - 1) * dilation;
    
    __shared__ scalar_t shared_input[TILE_H][TILE_W];
    
    const int ow = blockIdx.x * BLOCK_W + threadIdx.x;
    const int oh = blockIdx.y * BLOCK_H + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;
    
    // Load input tile into shared memory
    const int base_ih = blockIdx.y * BLOCK_H * stride - padding;
    const int base_iw = blockIdx.x * BLOCK_W * stride - padding;
    
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    
    #pragma unroll
    for (int dy = threadIdx.y; dy < TILE_H; dy += BLOCK_H) {
        #pragma unroll
        for (int dx = threadIdx.x; dx < TILE_W; dx += BLOCK_W) {
            int ih = base_ih + dy;
            int iw = base_iw + dx;
            shared_input[dy][dx] = (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) 
                                 ? input_channel[ih * input_width + iw]
                                 : -std::numeric_limits<scalar_t>::infinity();
        }
    }
    
    __syncthreads();
    
    if (ow >= output_width || oh >= output_height || b >= batch_size || c >= channels)
        return;
        
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    const int tile_start_h = threadIdx.y * stride;
    const int tile_start_w = threadIdx.x * stride;
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            const int sh = tile_start_h + kh * dilation;
            const int sw = tile_start_w + kw * dilation;
            max_val = max(max_val, shared_input[sh][sw]);
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
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_shared_kernel<scalar_t><<<blocks, threads>>>(
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