#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3  // Assumed fixed 3x3 kernel for loop unrolling
#define SHARED_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)

// This kernel uses shared memory to load input tiles and kernel weights, and employs
// #pragma unroll to unroll the inner loops for the kernel window, reducing loop overhead.
// It assumes square input, square kernel (3x3), stride and padding, with groups == 1.

__global__ void conv2d_unroll_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {

    // Each block is mapped to a tile of the output for a specific (batch, out_channel) pair.
    int b_oc = blockIdx.z; // blockIdx.z ranges over (batch_size * out_channels)
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;

    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx + tx; // output column index
    int y = by + ty; // output row index

    float sum = 0.0f;

    // Shared memory for input tile and kernel weights
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];

    // Loop over all input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load kernel weights for current (oc, ic) into shared memory
        if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
            // Weight layout: [out_channels, in_channels, KERNEL_SIZE, KERNEL_SIZE]
            int w_idx = ((oc * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx;
            shared_weight[ty][tx] = weight[w_idx];
        }

        // Load input tile for current (b, ic) into shared memory
        // Each block loads a tile of size SHARED_SIZE x SHARED_SIZE
        for (int i = ty; i < SHARED_SIZE; i += BLOCK_SIZE) {
            for (int j = tx; j < SHARED_SIZE; j += BLOCK_SIZE) {
                int in_y = by + i - padding;
                int in_x = bx + j - padding;
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    shared_input[i][j] = input[((b * in_channels + ic) * input_height + in_y) * input_width + in_x];
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute convolution for this input channel using unrolled loops
        if (x < output_width && y < output_height) {
            #pragma unroll
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    sum += shared_input[ty + ky][tx + kx] * shared_weight[ky][kx];
                }
            }
        }
        __syncthreads();
    }

    // Write the computed sum to the output tensor
    if (x < output_width && y < output_height) {
        int o_idx = ((b * out_channels + oc) * output_height + y) * output_width + x;
        output[o_idx] = sum;
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
    TORCH_CHECK(weight.size(2) == KERNEL_SIZE && weight.size(3) == KERNEL_SIZE,
                "This implementation assumes a 3x3 kernel");
    TORCH_CHECK(groups == 1, "groups != 1 is not supported by this kernel");

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
                batch_size * out_channels);

    conv2d_unroll_kernel<<<blocks, threads>>>(
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
        padding);

    cudaDeviceSynchronize();

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with loop unrolling and shared memory");
}
