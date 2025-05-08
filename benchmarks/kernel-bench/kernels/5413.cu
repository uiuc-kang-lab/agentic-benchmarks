#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int const_params[5]; // Hold kernel_size, stride, padding, dilation, in terms of setup

// Update the kernel to use constant memory for fixed parameters
template <typename scalar_t>
__global__ void max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // 2D thread indexing
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;

    if (ox >= output_width || oy >= output_height) return;

    // Calculate output index
    const int output_idx = ((b * channels + c) * output_height + oy) * output_width + ox;
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    #pragma unroll
    for (int kh = 0; kh < const_params[0]; kh++) {
        const int ih = oy * const_params[1] - const_params[2] + kh * const_params[3];
        if (ih >= 0 && ih < input_height) {
            #pragma unroll
            for (int kw = 0; kw < const_params[0]; kw++) {
                const int iw = ox * const_params[1] - const_params[2] + kw * const_params[3];
                if (iw >= 0 && iw < input_width) {
                    const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }

    output[output_idx] = max_val;
}

// Host function to launch the CUDA kernel with setup of constant memory
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

    // Setup constant memory
    int h_const_params[5] = {kernel_size, stride, padding, dilation, 0};
    cudaMemcpyToSymbol(const_params, h_const_params, sizeof(int) * 4);

    // 2D block configuration
    dim3 threads(32, 8);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
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
            output_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA)");
}