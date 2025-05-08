#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7
#define BLOCK_SIZE 16

__global__ void conv2d_shared_cache_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    __shared__ float shared_weight[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * BLOCK_SIZE];
    __shared__ float shared_input[TILE_SIZE * TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int out_x = bx * BLOCK_SIZE + tx;
    const int out_y = by * BLOCK_SIZE + ty;
    const int out_c = bz;

    // Pre-compute group information
    const int group_out_channels = out_channels / groups;
    const int group = out_c / group_out_channels;
    const int in_channels_per_group = in_channels / groups;

    // Load weights into shared memory
    const int weight_load_idx = ty * BLOCK_SIZE + tx;
    if (weight_load_idx < kernel_height * kernel_width * in_channels_per_group) {
        const int kh = weight_load_idx / (kernel_width * in_channels_per_group);
        const int kw = (weight_load_idx / in_channels_per_group) % kernel_width;
        const int ic = weight_load_idx % in_channels_per_group;
        shared_weight[weight_load_idx] = weight[((out_c * in_channels_per_group + ic) * kernel_height + kh) * kernel_width + kw];
    }
    __syncthreads();

    if (out_x >= out_width || out_y >= out_height) return;

    float sum = 0.0f;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            const int input_channel = group * in_channels_per_group + ic;
            
            // Load input tile into shared memory
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int in_y = out_y * stride - padding + kh * dilation;
                    const int in_x = out_x * stride - padding + kw * dilation;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        const int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        const int shared_idx = (ty * kernel_height + kh) * BLOCK_SIZE + (tx * kernel_width + kw);
                        if (shared_idx < TILE_SIZE * TILE_SIZE) {
                            shared_input[shared_idx] = input[input_idx];
                        }
                    }
                }
            }
            __syncthreads();

            // Compute convolution using shared memory
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int in_y = out_y * stride - padding + kh * dilation;
                    const int in_x = out_x * stride - padding + kw * dilation;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        const int weight_idx = (ic * kernel_height + kh) * kernel_width + kw;
                        const int shared_idx = (ty * kernel_height + kh) * BLOCK_SIZE + (tx * kernel_width + kw);
                        if (shared_idx < TILE_SIZE * TILE_SIZE) {
                            sum += shared_input[shared_idx] * shared_weight[weight_idx];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (bias != nullptr) {
        sum += bias[out_c];
    }

    if (out_x < out_width && out_y < out_height) {
        for (int b = 0; b < batch_size; ++b) {
            const int output_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
            output[output_idx] = sum;
        }
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

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_size(
        (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
        out_channels
    );

    const size_t shared_memory_size = 
        (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * BLOCK_SIZE + TILE_SIZE * TILE_SIZE) * sizeof(float);

    conv2d_shared_cache_kernel<<<grid_size, block_size, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory Weight Caching");
}