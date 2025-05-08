#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <stdexcept>

// Maximum allowed pooling window size (e.g., for a kernel up to 9x9, 81 elements)
#define MAX_POOL_SIZE 81

// Structure to hold pooling offset (row and column offsets)
struct PoolOffset {
    int dr;
    int dc;
};

// Constant memory to store precomputed pooling offsets
__constant__ PoolOffset d_pool_offsets[MAX_POOL_SIZE];
__constant__ int d_pool_offsets_count;

// CUDA kernel using constant memory for pooling offsets
// It computes max pooling over a 2D input tensor.
template <typename scalar_t>
__global__ void max_pool2d_constmem_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding
) {
    // Compute the output coordinates
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;  // Combined index for batch and channel
    const int b = bc / channels;
    const int c = bc % channels;

    if (b >= batch_size || c >= channels || oh >= output_height || ow >= output_width)
        return;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t* input_channel = input + (b * channels + c) * input_height * input_width;
    
    // Precompute the top-left starting indices in the input for this pooling window
    const int base_ih = oh * stride - padding;
    const int base_iw = ow * stride - padding;

    // Loop over the precomputed offsets stored in constant memory
    for (int i = 0; i < d_pool_offsets_count; i++) {
        int ih = base_ih + d_pool_offsets[i].dr;
        int iw = base_iw + d_pool_offsets[i].dc;
        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            max_val = max(max_val, input_channel[ih * input_width + iw]);
        }
    }

    output[(b * channels + c) * output_height * output_width + oh * output_width + ow] = max_val;
}

// Host function for the CUDA max pooling forward pass
// It precomputes pooling offsets (taking dilation into account) and stores them in constant memory
// before launching the kernel.

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

    // Precompute pooling offsets (for each element in the kernel window)
    int count = kernel_size * kernel_size;
    if (count > MAX_POOL_SIZE) {
        throw std::runtime_error("Kernel size too large for constant memory pool offsets");
    }
    PoolOffset h_pool_offsets[MAX_POOL_SIZE];
    for (int r = 0; r < kernel_size; r++) {
        for (int c = 0; c < kernel_size; c++) {
            h_pool_offsets[r * kernel_size + c].dr = r * dilation;
            h_pool_offsets[r * kernel_size + c].dc = c * dilation;
        }
    }

    // Copy the computed offsets to constant memory
    cudaMemcpyToSymbol(d_pool_offsets, h_pool_offsets, count * sizeof(PoolOffset));
    cudaMemcpyToSymbol(d_pool_offsets_count, &count, sizeof(int));

    // Launch kernel with a 2D configuration for spatial dimensions and combined batch-channel dimension
    const dim3 threads(32, 8);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool2d_cuda_forward", ([&] {
        max_pool2d_constmem_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            stride,
            padding
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_forward, "Max Pool 2D forward (CUDA) with constant memory offsets");
}
