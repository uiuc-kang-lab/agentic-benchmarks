#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile dimensions for output spatial block
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Custom CUDA kernel for 2D convolution with coalesced memory accesses
__global__ void conv2d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int N, int C, int H, int W,
                              int out_channels,
                              int kernel_h, int kernel_w,
                              int out_H, int out_W,
                              int stride, int padding, int dilation,
                              int groups, int group_in_channels) {
    // Each block in grid.z corresponds to one (n, oc) pair
    int n = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;

    // Determine the group index for the current output channel
    int group_idx = oc / (out_channels / groups);

    // Compute output spatial coordinates from block indices and thread indices
    int ow = blockIdx.x * TILE_WIDTH + threadIdx.x; // horizontal index
    int oh = blockIdx.y * TILE_HEIGHT + threadIdx.y; // vertical index

    if (n >= N || oc >= out_channels || oh >= out_H || ow >= out_W) return;

    float sum = 0.0f;
    int in_channel_start = group_idx * group_in_channels;

    // Loop over the relevant input channels and kernel window
    #pragma unroll
    for (int ic = 0; ic < group_in_channels; ++ic) {
        int input_channel = in_channel_start + ic;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = oh * stride - padding + kh * dilation;
            if (in_y < 0 || in_y >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = ow * stride - padding + kw * dilation;
                if (in_x < 0 || in_x >= W) continue;
                // Compute input index in NCHW layout
                int input_idx = n * (C * H * W) + input_channel * (H * W) + in_y * W + in_x;
                // Compute weight index: weights are stored as [out_channels, group_in_channels, kernel_h, kernel_w]
                int weight_idx = oc * (group_in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write the result to the output tensor in NCHW layout
    int output_idx = n * (out_channels * out_H * out_W) + oc * (out_H * out_W) + oh * out_W + ow;
    output[output_idx] = sum;
}

// Forward function that sets up and launches the convolution kernel
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // Retrieve input dimensions (N, C, H, W)
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Retrieve weight dimensions: [out_channels, group_in_channels, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int group_in_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Compute output spatial dimensions
    int out_H = (H + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_W = (W + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::zeros({N, out_channels, out_H, out_W}, input.options());

    // Configure block and grid dimensions to ensure coalesced accesses
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_W + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_H + TILE_HEIGHT - 1) / TILE_HEIGHT,
              N * out_channels); // each grid z corresponds to a (n, oc) pair

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_kernel<<<grid, block, 0, stream>>>(input_ptr, weight_ptr, bias_ptr, output_ptr,
                                               N, C, H, W,
                                               out_channels,
                                               kernel_h, kernel_w,
                                               out_H, out_W,
                                               stride, padding, dilation,
                                               groups, group_in_channels);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with coalesced memory access");
}
