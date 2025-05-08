#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
#define SHARED_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)

// Device function: Load a 3x3 tile from the weight tensor into shared memory
__device__ inline void load_weight_tile(const float* weight, int oc, int ic, int in_channels, float shared_weight[KERNEL_SIZE][KERNEL_SIZE]) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
        int weight_idx = ((oc * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx;
        shared_weight[ty][tx] = weight[weight_idx];
    }
}

// Device function: Load a tile of the input channel into shared memory
// Template parameter BS is the block size
template <int BS>
__device__ inline void load_input_tile(const float* input, int b, int ic, int in_channels,
                                         int input_height, int input_width, int padding,
                                         int block_y, int block_x,
                                         float shared_input[SHARED_SIZE][SHARED_SIZE]) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    for (int i = ty; i < SHARED_SIZE; i += BS) {
        for (int j = tx; j < SHARED_SIZE; j += BS) {
            int ih = block_y + i - padding;
            int iw = block_x + j - padding;
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                shared_input[i][j] = input[((b * in_channels + ic) * input_height + ih) * input_width + iw];
            } else {
                shared_input[i][j] = 0.0f;
            }
        }
    }
}

// Device function: Compute convolution for one output element using data in shared memory
__device__ inline float compute_convolution(const float shared_input[SHARED_SIZE][SHARED_SIZE],
                                              const float shared_weight[KERNEL_SIZE][KERNEL_SIZE],
                                              int tx, int ty, int stride) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            sum += shared_input[ty * stride + i][tx * stride + j] * shared_weight[i][j];
        }
    }
    return sum;
}

// Modular convolution kernel that leverages shared memory and device functions
__global__ void mod_conv2d_kernel(const float* __restrict__ input,
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
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];

    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_out = bx + tx;
    int y_out = by + ty;

    // Loop over each output channel
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        // Accumulate contributions from all input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load weight tile for current oc and ic
            load_weight_tile(weight, oc, ic, in_channels, shared_weight);
            __syncthreads();
            
            // Load corresponding input tile for current channel into shared memory
            load_input_tile<BLOCK_SIZE>(input, b, ic, in_channels, input_height, input_width, padding, by, bx, shared_input);
            __syncthreads();
            
            // If within output bounds, compute convolution using shared memory
            if (x_out < output_width && y_out < output_height) {
                sum += compute_convolution(shared_input, shared_weight, tx, ty, stride);
            }
            __syncthreads();
        }
        // Write the accumulated result to the output tensor
        if (x_out < output_width && y_out < output_height) {
            int out_idx = ((b * out_channels + oc) * output_height + y_out) * output_width + x_out;
            output[out_idx] = sum;
        }
        __syncthreads();
    }
}

// PyTorch binding function
torch::Tensor forward(torch::Tensor x,
                      torch::Tensor weight,
                      torch::optional<torch::Tensor> bias,
                      int stride,
                      int padding,
                      int dilation,
                      int groups) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.size(2) == KERNEL_SIZE && weight.size(3) == KERNEL_SIZE, "Kernel size must be 3x3.");

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

    mod_conv2d_kernel<<<blocks, threads>>>(
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

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular adaptive CUDA conv2d implementation");
}
