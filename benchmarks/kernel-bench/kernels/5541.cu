#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    constexpr int TILE_SIZE = 16;
    __shared__ scalar_t shared_input[TILE_SIZE + 4][TILE_SIZE + 4];  // +4 for maximum kernel size padding

    const int tile_x = blockIdx.x * TILE_SIZE;
    const int tile_y = blockIdx.y * TILE_SIZE;
    const int channel = blockIdx.z % channels;
    const int batch = blockIdx.z / channels;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread processes multiple elements for better workload distribution
    #pragma unroll
    for (int y_offset = 0; y_offset < TILE_SIZE; y_offset += blockDim.y) {
        const int oh = tile_y + ty + y_offset;
        #pragma unroll
        for (int x_offset = 0; x_offset < TILE_SIZE; x_offset += blockDim.x) {
            const int ow = tile_x + tx + x_offset;

            if (oh < output_height && ow < output_width) {
                scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

                const int ih_start = oh * stride - padding;
                const int iw_start = ow * stride - padding;

                // Input offset for the current batch and channel
                const int input_offset = (batch * channels * input_height * input_width) +
                                       (channel * input_height * input_width);

                // Collaborative loading of input data into shared memory
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    const int ih = ih_start + kh * dilation;
                    if (ih >= 0 && ih < input_height) {
                        #pragma unroll
                        for (int kw = 0; kw < kernel_size; kw++) {
                            const int iw = iw_start + kw * dilation;
                            if (iw >= 0 && iw < input_width) {
                                const scalar_t val = __ldg(&input[input_offset + ih * input_width + iw]);
                                max_val = max(max_val, val);
                            }
                        }
                    }
                }

                // Write output
                if (max_val > -std::numeric_limits<scalar_t>::infinity()) {
                    const int output_idx = (batch * channels * output_height * output_width) +
                                         (channel * output_height * output_width) +
                                         (oh * output_width) + ow;
                    output[output_idx] = max_val;
                }
            }
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

    // Configure grid and block dimensions for balanced workload
    const dim3 threads(8, 8);  // 64 threads per block
    const dim3 blocks(
        (output_width + 15) / 16,   // Ceil division by TILE_SIZE
        (output_height + 15) / 16,  // Ceil division by TILE_SIZE
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
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