#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3

// This kernel assumes stride == 1, dilation == 1, groups == 1 and a 3x3 kernel.
// It loads input tiles and weight tiles into shared memory using a 1D loop to ensure global memory coalescing.

__global__ void conv2d_coalesced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding) {

    // For stride == 1, the shared tile dimension is BLOCK_SIZE + KERNEL_SIZE - 1.
    const int tile_dim = BLOCK_SIZE + KERNEL_SIZE - 1; 
    __shared__ float shared_input[tile_dim * tile_dim];
    __shared__ float shared_weight[KERNEL_SIZE * KERNEL_SIZE];

    // Compute output spatial coordinates for this thread
    int out_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int b = blockIdx.z;

    // Compute the top-left coordinate of the input tile
    int in_x_origin = blockIdx.x * BLOCK_SIZE - padding;
    int in_y_origin = blockIdx.y * BLOCK_SIZE - padding;

    // Flatten thread index within the block for coalesced loads
    int t_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    // Loop over output channels
    for (int oc = 0; oc < out_channels; ++oc) {
        float result = 0.0f;
        // Sum over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load the input tile for the current (b, ic) channel into shared memory.
            int total_tile = tile_dim * tile_dim;
            for (int index = t_idx; index < total_tile; index += threads_per_block) {
                int i = index / tile_dim;
                int j = index % tile_dim;
                int in_i = in_y_origin + i;
                int in_j = in_x_origin + j;
                if (in_i >= 0 && in_i < input_height && in_j >= 0 && in_j < input_width) {
                    shared_input[index] = input[((b * in_channels + ic) * input_height + in_i) * input_width + in_j];
                } else {
                    shared_input[index] = 0.0f;
                }
            }
            
            // Load the 3x3 kernel for the current (oc, ic) pair into shared memory
            int total_weight = KERNEL_SIZE * KERNEL_SIZE;
            for (int index = t_idx; index < total_weight; index += threads_per_block) {
                shared_weight[index] = weight[((oc * in_channels + ic) * total_weight) + index];
            }
            
            __syncthreads();
            
            // Each thread computes the convolution for its output pixel if within bounds
            if (out_y < output_height && out_x < output_width) {
                int local_y = threadIdx.y;
                int local_x = threadIdx.x;
                float sum = 0.0f;
                #pragma unroll
                for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                    #pragma unroll
                    for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                        int shared_index = (local_y + ki) * tile_dim + (local_x + kj);
                        sum += shared_input[shared_index] * shared_weight[ki * KERNEL_SIZE + kj];
                    }
                }
                result += sum;
            }
            __syncthreads();
        }
        // Write the result to global memory in a coalesced manner
        if (out_y < output_height && out_x < output_width) {
            output[((b * out_channels + oc) * output_height + out_y) * output_width + out_x] = result;
        }
        __syncthreads();
    }
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    // This kernel only supports stride==1, dilation==1, groups==1 and 3x3 kernels
    TORCH_CHECK(stride == 1, "Only stride==1 is supported in conv2d_coalesced_kernel");
    TORCH_CHECK(dilation == 1, "Only dilation==1 is supported in conv2d_coalesced_kernel");
    TORCH_CHECK(groups == 1, "Only groups==1 is supported in conv2d_coalesced_kernel");
    TORCH_CHECK(weight.size(2) == KERNEL_SIZE && weight.size(3) == KERNEL_SIZE, "Only 3x3 kernel supported");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    int output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, x.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size);

    conv2d_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding
    );

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for conv2d with memory coalescing");
}
