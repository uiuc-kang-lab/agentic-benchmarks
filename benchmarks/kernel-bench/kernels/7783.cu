#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory tiling to load input patches once and reuse them for convolution.
// Each block computes an output tile for a single (batch, output channel) pair, ensuring even workload distribution.

__global__ void conv2d_shared_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Tile dimensions for the output
    const int TILE_W = 16;
    const int TILE_H = 16;

    // Determine which (batch, output_channel) this block is processing
    int b = blockIdx.z / out_channels;     // b is the batch index
    int oc = blockIdx.z % out_channels;      // oc is the output channel

    // Starting indices for this output tile
    int out_tile_x = blockIdx.x * TILE_W;
    int out_tile_y = blockIdx.y * TILE_H;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global output coordinates
    int out_x = out_tile_x + tx;
    int out_y = out_tile_y + ty;

    // Dimensions for the shared memory tile
    // Each output tile requires a corresponding input patch of size:
    // sm_width = TILE_W * stride + (kernel_width - 1) * dilation
    // sm_height = TILE_H * stride + (kernel_height - 1) * dilation
    int sm_width = TILE_W * stride + (kernel_width - 1) * dilation;
    int sm_height = TILE_H * stride + (kernel_height - 1) * dilation;

    // Declare dynamic shared memory
    extern __shared__ float smem[];

    // Top-left corner in the input corresponding to this output tile
    int in_tile_x = out_tile_x * stride - padding;
    int in_tile_y = out_tile_y * stride - padding;

    // Compute linear thread id for cooperative loading
    int num_threads = blockDim.x * blockDim.y;
    int thread_id = ty * blockDim.x + tx;

    // Determine group information
    int group_out_channels = out_channels / groups;
    int group = oc / group_out_channels;
    int in_channels_per_group = in_channels / groups;

    float sum = 0.0f;

    // Loop over the input channels for this group
    for (int ic_idx = 0; ic_idx < in_channels_per_group; ic_idx++) {
        int ic = group * in_channels_per_group + ic_idx;
        const float* input_channel = input + ((b * in_channels + ic) * in_height * in_width);

        // Cooperative loading of the input patch into shared memory
        int smem_size = sm_width * sm_height;
        for (int i = thread_id; i < smem_size; i += num_threads) {
            int sm_x = i % sm_width;
            int sm_y = i / sm_width;
            int gx = in_tile_x + sm_x;
            int gy = in_tile_y + sm_y;
            float val = 0.0f;
            if (gx >= 0 && gx < in_width && gy >= 0 && gy < in_height) {
                val = input_channel[gy * in_width + gx];
            }
            smem[i] = val;
        }
        __syncthreads();

        // Compute convolution for the current input channel
        if (out_x < out_width && out_y < out_height) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int sm_x = tx * stride + kw * dilation;
                    int sm_y = ty * stride + kh * dilation;
                    if (sm_x < sm_width && sm_y < sm_height) {
                        float in_val = smem[sm_y * sm_width + sm_x];
                        // Weight layout: [out_channels, in_channels_per_group, kernel_height, kernel_width]
                        int weight_idx = (((oc * in_channels_per_group + ic_idx) * kernel_height + kh) * kernel_width) + kw;
                        float w_val = weight[weight_idx];
                        sum += in_val * w_val;
                    }
                }
            }
        }
        __syncthreads();  // Ensure all threads are done before reusing shared memory
    }

    // Write the result if within bounds
    if (out_x < out_width && out_y < out_height) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    // Define tile dimensions
    const int TILE_W = 16;
    const int TILE_H = 16;

    // Grid dimensions: spatial tiles and one block per (batch, out_channel)
    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_width + TILE_W - 1) / TILE_W,
              (out_height + TILE_H - 1) / TILE_H,
              batch_size * out_channels);

    // Shared memory size calculation
    int sm_width = TILE_W * stride + (kernel_width - 1) * dilation;
    int sm_height = TILE_H * stride + (kernel_height - 1) * dilation;
    size_t shared_memory_size = sm_width * sm_height * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    conv2d_shared_tiled_kernel<<<grid, block, shared_memory_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory Tiling and Balanced Workload Distribution");
}
