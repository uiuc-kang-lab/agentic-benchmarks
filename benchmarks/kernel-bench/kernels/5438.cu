#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool2d_kernel_shared(
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
    const int dilation
) {
    extern __shared__ char shared_memory[];
    scalar_t* shared_input = reinterpret_cast<scalar_t*>(shared_memory);

    const int tile_size = 16;
    const int input_tile_size = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * tile_size;
    const int by = blockIdx.y * tile_size;
    const int b = blockIdx.z / channels;
    const int c = blockIdx.z % channels;

    const int input_start_x = bx * stride - padding;
    const int input_start_y = by * stride - padding;

    for (int y = ty; y < input_tile_size; y += blockDim.y) {
        for (int x = tx; x < input_tile_size; x += blockDim.x) {
            const int ih = input_start_y + y;
            const int iw = input_start_x + x;
            
            const int shared_idx = y * input_tile_size + x;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width + iw;
                shared_input[shared_idx] = __ldg(&input[input_idx]);
            } else {
                shared_input[shared_idx] = -std::numeric_limits<scalar_t>::infinity();
            }
        }
    }
    __syncthreads();

    const int out_x = bx + tx;
    const int out_y = by + ty;
    
    if (out_x < output_width && out_y < output_height) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int sh_y = ty * stride + kh * dilation;
                const int sh_x = tx * stride + kw * dilation;
                const int shared_idx = sh_y * input_tile_size + sh_x;
                max_val = max(max_val, shared_input[shared_idx]);
            }
        }

        if (out_x < output_width && out_y < output_height) {
            const int output_idx = b * (channels * output_height * output_width) +
                                 c * (output_height * output_width) +
                                 out_y * output_width + out_x;
            output[output_idx] = max_val;
        }
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

    const int tile_size = 16;
    const int input_tile_size = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    const int shared_memory_size = input_tile_size * input_tile_size * sizeof(float);

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + tile_size - 1) / tile_size,
        (output_height + tile_size - 1) / tile_size,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel_shared<scalar_t><<<blocks, threads, shared_memory_size>>>(
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