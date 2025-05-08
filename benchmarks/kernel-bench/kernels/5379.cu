#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Define constant memory for kernel parameters
__constant__ int const_params[4]; // [kernel_size, stride, padding, dilation]

// Optimized kernel combining coalesced memory access and constant memory
template <typename scalar_t>
__global__ void max_pool2d_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    const int kernel_size = const_params[0];
    const int stride = const_params[1];
    const int padding = const_params[2];
    const int dilation = const_params[3];

    // Calculate output spatial coordinates
    int ow = threadIdx.x + blockIdx.x * blockDim.x;
    int oh = threadIdx.y + blockIdx.y * blockDim.y;

    // Each block in z-dimension corresponds to one (batch, channel) pair
    int bc = blockIdx.z;
    if (ow >= output_width || oh >= output_height)
        return;

    int b = bc / channels;
    int c = bc % channels;

    // Compute the flat output index (row-major order: b, c, oh, ow)
    int output_idx = ((b * channels + c) * output_height + oh) * output_width + ow;
    int base_input_offset = ((b * channels + c) * input_height) * input_width;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    // Determine the top-left corner of the pooling window
    int hstart = oh * stride - padding;
    int wstart = ow * stride - padding;

    // Iterate over the pooling window
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = hstart + kh * dilation;
        if (ih < 0 || ih >= input_height) continue;
        for (int kw = 0; kw < kernel_size; kw++) {
            int iw = wstart + kw * dilation;
            if (iw < 0 || iw >= input_width) continue;
            int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
            // Use __ldg for read-only memory access to leverage the texture cache
            scalar_t val = __ldg(&input[input_idx]);
            max_val = max(max_val, val);
        }
    }

    output[output_idx] = max_val;
}

// Host function that sets up the kernel launch parameters using a 2D block and 3D grid
// to align memory accesses so that threads in a warp write to consecutive locations

torch::Tensor max_pool2d_optimized_cuda_forward(
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
    const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int params[4] = {kernel_size, stride, padding, dilation};
    cudaMemcpyToSymbol(const_params, params, sizeof(params));

    // Use a 2D block: 32 threads in x (matches warp size) and 8 in y for a total of 256 threads
    dim3 block(32, 8);
    // Grid dimensions: cover the spatial dimensions and handle all (batch, channel) pairs in grid.z
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_pool2d_optimized_cuda_forward", ([&] {
        max_pool2d_optimized_kernel<scalar_t><<<grid, block>>>(
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
    m.def("forward", &max_pool2d_optimized_cuda_forward, "Optimized Max Pool 2D forward (CUDA)");
}
