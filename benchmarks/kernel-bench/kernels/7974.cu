#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes one batch's convolution output for a given channel.
// It iterates over the input channels and kernel elements to accumulate the convolution sum.
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_height,
    int out_width) {

    // Each thread computes one output pixel in the spatial dimensions for a given output channel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;  // output channel index

    if (row < out_height && col < out_width && channel < out_channels) {
        float sum = 0.0f;
        // Loop over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Loop over kernel height and width
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_row = row * stride - padding + kh * dilation;
                    int in_col = col * stride - padding + kw * dilation;
                    if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
                        int input_idx = (ic * in_height + in_row) * in_width + in_col;
                        int weight_idx = ((channel * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        int output_idx = (channel * out_height + row) * out_width + col;
        output[output_idx] = sum;
    }
}


// Host function that launches the convolution kernel for each batch concurrently using separate CUDA streams to overlap computation with any memory operations.
// Note: This implementation currently supports groups == 1.

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

    TORCH_CHECK(groups == 1, "stream_pipelined_conv2d only supports groups == 1");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // square kernel assumed
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    // Configure a 2D grid for the spatial output and iterate over channels via grid.z
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              out_channels);

    // Create a CUDA stream per batch to allow overlapping execution
    std::vector<cudaStream_t> streams(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        cudaStreamCreate(&streams[b]);
        conv2d_kernel<<<grid, block, 0, streams[b]>>>(
            x[b].data_ptr<float>(),
            weight.data_ptr<float>(),
            output[b].data_ptr<float>(),
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            out_height,
            out_width
        );
    }
    
    // Synchronize and destroy the streams
    for (int b = 0; b < batch_size; ++b) {
        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    if (bias.has_value()) {
        // Bias addition can be done on the default stream and is relatively fast
        output += bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream pipelined 2D convolution with overlapped computation and memory transfers");
}
