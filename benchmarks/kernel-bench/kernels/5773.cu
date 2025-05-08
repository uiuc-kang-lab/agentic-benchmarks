#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel uses a 3D grid mapping: 
//     - X dimension: output width
//     - Y dimension: output height
//     - Z dimension: combined batch and channel index (b * channels + c)
// Each thread computes one output element.

template <typename scalar_t>
__global__ void max_pool2d_kernel_3d(
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
    // Calculate output indices using 3D grid and block indexing
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z * blockDim.z + threadIdx.z;  // bc represents combined index: b * channels + c

    if (ow < output_width && oh < output_height && bc < batch_size * channels) {
        int b = bc / channels;
        int c = bc % channels;
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

        // Iterate over the pooling window
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    int input_idx = b * (channels * input_height * input_width) +
                                    c * (input_height * input_width) +
                                    ih * input_width +
                                    iw;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
        int output_idx = bc * (output_height * output_width) + oh * output_width + ow;
        output[output_idx] = max_val;
    }
}

// Host function that sets up grid and block dimensions using 3D indexing
torch::Tensor max_pool2d_cuda_forward_3d(
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

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Define 3D block dimensions; adjust as appropriate for your GPU and problem size
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (output_width  + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        ((batch_size * channels) + block.z - 1) / block.z
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cuda_forward_3d", ([&] {
        max_pool2d_kernel_3d<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_pool2d_cuda_forward_3d, "Max Pool 2D forward with 3D grid indexing (CUDA)");
}
