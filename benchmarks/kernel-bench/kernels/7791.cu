#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that processes a single batch element for 2D convolution using a dedicated CUDA stream.
// Each kernel launch computes the convolution for one batch index, mapping spatial positions and output channels to threads.
__global__ void conv2d_stream_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr if not provided
    float* __restrict__ output,
    int batch_idx,
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

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;  // Each block in z-dimension corresponds to an output channel

    if (w < out_width && h < out_height && oc < out_channels) {
        float sum = 0.0f;
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        for (int c = 0; c < in_channels_per_group; ++c) {
            int input_channel = group * in_channels_per_group + c;
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int in_y = h * stride - padding + kh * dilation;
                    int in_x = w * stride - padding + kw * dilation;
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((batch_idx * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                        int weight_idx = ((oc * in_channels_per_group + c) * kernel_height + kh) * kernel_width + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = ((batch_idx * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
    }
}

// Forward function that splits the batch dimension and launches an independent kernel in its own CUDA stream.
// This design enables overlapping of kernel computation with potential asynchronous memory transfers (e.g., device-host copies), 
// thereby pipelining operations to improve overall throughput on NVIDIA H100.

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

    // Calculate output spatial dimensions using standard convolution formula
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width  = (in_width  + 2 * padding - dilation * (kernel_width - 1)  - 1) / stride + 1;

    auto options = x.options();
    torch::Tensor output = torch::empty({batch_size, out_channels, out_height, out_width}, options);

    // Set up block and grid dimensions for each kernel launch
    dim3 block(16, 16, 1);
    dim3 grid((out_width  + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              out_channels);

    // Create a CUDA stream for each batch element to overlap kernel execution with memory transfers
    std::vector<cudaStream_t> streams(batch_size);

    const float* input_ptr  = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr   = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_ptr       = output.data_ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        cudaStreamCreate(&streams[b]);
        conv2d_stream_kernel<<<grid, block, 0, streams[b]>>>(
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            b,
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

        // If required, initiate asynchronous memory copies (e.g., device-to-host) here using the same stream
        // to overlap data transfers with computation for subsequent batches.
        // Example (commented out):
        // cudaMemcpyAsync(host_buffer + b * batch_output_size,
        //                 output_ptr + b * batch_output_size,
        //                 batch_output_size * sizeof(float),
        //                 cudaMemcpyDeviceToHost,
        //                 streams[b]);
    }

    // Synchronize all streams and destroy them
    for (int b = 0; b < batch_size; ++b) {
        cudaStreamSynchronize(streams[b]);
        cudaStreamDestroy(streams[b]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Asynchronous Streaming and Overlapped Memory Transfers");
}
