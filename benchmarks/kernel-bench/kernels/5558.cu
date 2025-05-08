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
    const int TILE_SIZE = 32;
    __shared__ scalar_t shared_input[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x * blockDim.x + tid;
    
    if (output_idx >= batch_size * channels * output_height * output_width) return;

    const int ow = output_idx % output_width;
    const int oh = (output_idx / output_width) % output_height;
    const int c = (output_idx / (output_width * output_height)) % channels;
    const int b = output_idx / (output_width * output_height * channels);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Calculate input region boundaries
    const int start_ih = oh * stride - padding;
    const int start_iw = ow * stride - padding;
    const int end_ih = start_ih + kernel_size * dilation;
    const int end_iw = start_iw + kernel_size * dilation;

    // Process input in tiles
    for (int tile_ih = start_ih; tile_ih < end_ih; tile_ih += TILE_SIZE) {
        for (int tile_iw = start_iw; tile_iw < end_iw; tile_iw += TILE_SIZE) {
            // Load tile into shared memory
            if (tid < TILE_SIZE) {
                for (int i = 0; i < TILE_SIZE; i++) {
                    const int ih = tile_ih + i;
                    const int iw = tile_iw + tid;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        const int input_idx = b * (channels * input_height * input_width) +
                                            c * (input_height * input_width) +
                                            ih * input_width + iw;
                        shared_input[i][tid] = input[input_idx];
                    } else {
                        shared_input[i][tid] = -std::numeric_limits<scalar_t>::infinity();
                    }
                }
            }
            __syncthreads();

            // Process the tile
            for (int kh = 0; kh < kernel_size; kh++) {
                const int ih = tile_ih + kh * dilation - start_ih;
                if (ih >= 0 && ih < TILE_SIZE) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        const int iw = tile_iw + kw * dilation - start_iw;
                        if (iw >= 0 && iw < TILE_SIZE) {
                            max_val = max(max_val, shared_input[ih][iw]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    output[output_idx] = max_val;
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