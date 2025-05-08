#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// CUDA kernel using shared memory tiling
__global__ void depthwise_conv2d_shared_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    // Each block processes one depthwise channel for one batch element.
    int n = blockIdx.z; // n in [0, batch_size*out_channels)
    int b = n / out_channels;
    int c = n % out_channels;
    int g = c / channels_per_group;
    int m = c % channels_per_group;

    // Determine the starting coordinates of the output tile this block is responsible for
    int tile_out_y = blockIdx.y * TILE_HEIGHT;
    int tile_out_x = blockIdx.x * TILE_WIDTH;

    // Compute the dimensions of the shared memory tile needed.
    // This tile covers the output region plus the necessary halo for the convolution
    int shared_h = (TILE_HEIGHT - 1) * stride_h + dilation_h * (kernel_h - 1) + 1;
    int shared_w = (TILE_WIDTH - 1) * stride_w + dilation_w * (kernel_w - 1) + 1;
    // Add padding to avoid bank conflicts (pad to 32-bit boundary)
    int shared_w_padded = (shared_w + 31) & ~31;

    // Compute the corresponding top-left coordinate in the input
    int in_tile_start_y = tile_out_y * stride_h - pad_h;
    int in_tile_start_x = tile_out_x * stride_w - pad_w;

    // Allocate shared memory (declared dynamically via extern)
    extern __shared__ float shmem[]; // size = shared_h * shared_w

    // Cooperative load of shared memory: each thread loads multiple elements if needed
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int shmem_size = shared_h * shared_w;
    for (int index = tid; index < shmem_size; index += total_threads) {
        int sh_y = index / shared_w;
        int sh_x = index % shared_w;
        int in_y = in_tile_start_y + sh_y;
        int in_x = in_tile_start_x + sh_x;
        float value = 0.0f;
        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
            int input_idx = ((b * in_channels + g) * in_h + in_y) * in_w + in_x;
            value = input[input_idx];
        }
        shmem[index] = value;
    }
    __syncthreads();

    // Compute output coordinate in the global output tensor
    int out_y = tile_out_y + threadIdx.y;
    int out_x = tile_out_x + threadIdx.x;

    if (out_y < out_h && out_x < out_w) {
        float sum = 0.0f;
        // Each thread computes convolution for its output coordinate using shared memory
        // The shared memory pointer offset for the top-left of the receptive field for this thread
        // is (threadIdx.y * stride_h, threadIdx.x * stride_w).
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int sh_y = threadIdx.y * stride_h + kh * dilation_h;
                int sh_x = threadIdx.x * stride_w + kw * dilation_w;
                float in_val = shmem[sh_y * shared_w + sh_x];
                int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                float w_val = weight[weight_idx];
                sum += in_val * w_val;
            }
        }
        if (bias != nullptr) {
            sum += bias[c];
        }
        int out_idx = ((b * out_channels + c) * out_h + out_y) * out_w + out_x;
        output[out_idx] = sum;
    }
}

// Forward function called from pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    // Calculate output dimensions
    int out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    // Set block and grid dimensions.
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
              batch_size * out_channels);

    // Compute shared memory size
    int shared_h = (TILE_HEIGHT - 1) * stride_h + dilation_h * (kernel_h - 1) + 1;
    int shared_w = (TILE_WIDTH - 1) * stride_w + dilation_w * (kernel_w - 1) + 1;
    size_t shared_mem_size = shared_h * shared_w * sizeof(float);

    depthwise_conv2d_shared_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward with shared memory (CUDA)");
}
