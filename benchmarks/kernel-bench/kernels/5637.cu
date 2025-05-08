#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Kernel that maps thread blocks to 2D spatial output and uses gridDim.z for batch and channel dimensions

template <typename scalar_t>
__global__ void grid2d_maxpool2d_kernel(
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
    // Derive batch and channel indices from blockIdx.z
    int bc_index = blockIdx.z;
    int c = bc_index % channels;
    int b = bc_index / channels;

    // Define tile dimensions for spatial mapping
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;

    // Calculate output spatial indices for this thread
    int ow = blockIdx.x * TILE_W + threadIdx.x;
    int oh = blockIdx.y * TILE_H + threadIdx.y;

    if (b < batch_size && c < channels && oh < output_height && ow < output_width) {
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Base pointer offset for the current (b, c) in the input tensor
        int input_base = b * channels * input_height * input_width + c * input_height * input_width;

        // Loop over the pooling window
        for (int kh = 0; kh < kernel_size; kh++) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = ow * stride - padding + kw * dilation;
                if (iw < 0 || iw >= input_width) continue;
                int input_idx = input_base + ih * input_width + iw;
                scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        // Write the result to output tensor
        int output_base = b * channels * output_height * output_width + c * output_height * output_width;
        output[output_base + oh * output_width + ow] = max_val;
    }
}


// Host function to launch the kernel with a 3D grid mapping

torch::Tensor grid2d_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Calculate output dimensions
    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    // Define tile dimensions
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    
    // Grid dimensions: x for width, y for height, z for (batch * channels)
    dim3 threads(TILE_W, TILE_H);
    dim3 blocks(
        (output_width + TILE_W - 1) / TILE_W,
        (output_height + TILE_H - 1) / TILE_H,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid2d_maxpool2d_cuda_forward", ([&] {
        grid2d_maxpool2d_kernel<scalar_t><<<blocks, threads>>>(
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &grid2d_maxpool2d_cuda_forward, "Max Pool 2D forward with 2D grid indexing (CUDA)");
}
