#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t, int BLOCK_SIZE_X = 32, int BLOCK_SIZE_Y = 8>
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
    const int TILE_SIZE_X = BLOCK_SIZE_X + (kernel_size - 1) * dilation;
    const int TILE_SIZE_Y = BLOCK_SIZE_Y + (kernel_size - 1) * dilation;
    __shared__ scalar_t shared_input[TILE_SIZE_Y][TILE_SIZE_X];

    const int ow = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int oh = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc % channels;

    // Calculate input tile origin
    const int tile_start_h = blockIdx.y * BLOCK_SIZE_Y * stride - padding;
    const int tile_start_w = blockIdx.x * BLOCK_SIZE_X * stride - padding;

    // Load input tile into shared memory
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    
    #pragma unroll
    for (int i = threadIdx.y; i < TILE_SIZE_Y; i += BLOCK_SIZE_Y) {
        #pragma unroll
        for (int j = threadIdx.x; j < TILE_SIZE_X; j += BLOCK_SIZE_X) {
            const int ih = tile_start_h + i;
            const int iw = tile_start_w + j;
            
            shared_input[i][j] = (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) 
                ? input_channel[ih * input_width + iw]
                : -std::numeric_limits<scalar_t>::infinity();
        }
    }
    
    __syncthreads();  // Single sync point after shared memory load

    if (ow >= output_width || oh >= output_height || b >= batch_size)
        return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const int base_h = threadIdx.y * stride;
    const int base_w = threadIdx.x * stride;

    if (kernel_size == 2) {
        #pragma unroll
        for (int kh = 0; kh < 2; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 2; kw++) {
                const int h = base_h + kh * dilation;
                const int w = base_w + kw * dilation;
                max_val = max(max_val, shared_input[h][w]);
            }
        }
    }
    else if (kernel_size == 3) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                const int h = base_h + kh * dilation;
                const int w = base_w + kw * dilation;
                max_val = max(max_val, shared_input[h][w]);
            }
        }
    }
    else {
        #pragma unroll 4
        for (int kh = 0; kh < kernel_size; kh++) {
            #pragma unroll 4
            for (int kw = 0; kw < kernel_size; kw++) {
                const int h = base_h + kh * dilation;
                const int w = base_w + kw * dilation;
                max_val = max(max_val, shared_input[h][w]);
            }
        }
    }

    if (ow < output_width && oh < output_height) {
        output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
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

    constexpr int BLOCK_SIZE_X = 32;
    constexpr int BLOCK_SIZE_Y = 8;

    const dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 blocks(
        (output_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (output_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
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