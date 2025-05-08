/*
 * Combined CUDA kernel for 1D convolution using constant memory for weights and bias,
 * together with pipelined streams for batch-level parallelism.
 * This approach leverages the low-latency constant memory for small weights/bias
 * and overlapping kernel execution for large batches using multiple streams.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Define maximum allowed sizes for constant memory storage
// Typically 64KB (16K floats) is available, here we use 15360 for weight and 1024 for bias
#define MAX_WEIGHT_SIZE 15360
#define MAX_BIAS_SIZE 1024

// Declare constant memory arrays for weight and bias
__constant__ float c_weight[MAX_WEIGHT_SIZE];
__constant__ float c_bias[MAX_BIAS_SIZE];

// Combined 1D convolution kernel using constant memory for weight/bias.
// This kernel processes a segment of the batch (sub-batch) provided by the host.
// Parameters:
//   x         : Input tensor segment pointer with shape [segmentB, in_channels, in_size]
//   output    : Output tensor segment pointer with shape [segmentB, out_channels, out_size]
//   segmentB  : Number of batches in this segment
//   in_channels, in_size: Input dimensions
//   out_channels, kernel_size, out_size: Convolution filter parameters
//   stride, dilation: Convolution parameters
//   use_bias  : Whether to add bias
__global__ void conv1d_const_streams_kernel(
    const float* __restrict__ x,
    float* output,
    int segmentB,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    bool use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = segmentB * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels; // local batch index within the segment

    float sum = 0.0f;
    int base = o * stride;
    for (int ic = 0; ic < in_channels; ++ic) {
        int x_batch_offset = b * (in_channels * in_size);
        int x_ic_offset = ic * in_size;
        for (int k = 0; k < kernel_size; ++k) {
            int pos = base + k * dilation;
            if (pos < in_size) {
                int x_index = x_batch_offset + x_ic_offset + pos;
                // Weight index: weights stored in constant memory
                int w_index = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                sum += x[x_index] * c_weight[w_index];
            }
        }
    }
    if (use_bias) {
        sum += c_bias[oc];
    }
    int out_index = b * (out_channels * out_size) + oc * out_size + o;
    output[out_index] = sum;
}

// Forward function that loads weight (and bias) into constant memory and launches the kernel
// using multiple CUDA streams for pipelined execution along the batch dimension.
// Parameters:
//   x: Input tensor of shape [B, in_channels, in_size]
//   weight: Filter tensor of shape [out_channels, in_channels, kernel_size]
//   bias: Optional bias tensor of shape [out_channels]
//   stride, dilation: Convolution parameters
//   num_streams: Number of CUDA streams to use (default=4)

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation,
    int num_streams = 4
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, in_channels, in_size)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channel mismatch between x and weight");

    bool use_bias = false;
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size must match number of output channels");
        use_bias = true;
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size computed");

    // Ensure that weight and bias fit into the constant memory limits
    TORCH_CHECK(weight.numel() <= MAX_WEIGHT_SIZE, "Weight tensor too large for constant memory");
    if (use_bias) {
        TORCH_CHECK(bias->numel() <= MAX_BIAS_SIZE, "Bias tensor too large for constant memory");
    }

    // Copy weight (and bias) from device memory to constant memory
    size_t weight_bytes = weight.numel() * sizeof(float);
    cudaMemcpyToSymbolAsync(c_weight, weight.data_ptr<float>(), weight_bytes, 0, cudaMemcpyDeviceToDevice, streams[i]);
    if (use_bias) {
        size_t bias_bytes = bias->numel() * sizeof(float);
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), bias_bytes, 0, cudaMemcpyDeviceToDevice);
    }

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Determine how many streams to use based on the batch size
    int streams_to_use = std::min(num_streams, B);
    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch dimension among available streams
    int batch_per_stream = (B + streams_to_use - 1) / streams_to_use;
    int threads = 256;
    
    for (int i = 0; i < streams_to_use; ++i) {
        int start_B = i * batch_per_stream;
        int end_B = std::min(start_B + batch_per_stream, B);
        if (start_B >= end_B) continue;
        int current_B = end_B - start_B;
        int total_elements_segment = current_B * out_channels * out_size;
        int blocks = (total_elements_segment + threads - 1) / threads;

        // Launch the kernel for this batch segment on its own CUDA stream
        conv1d_const_streams_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_data + start_B * in_channels * in_size,
            output_data + start_B * out_channels * out_size,
            current_B,
            in_channels,
            in_size,
            out_channels,
            kernel_size,
            out_size,
            stride,
            dilation,
            use_bias
        );
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < streams_to_use; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with combined constant memory and stream pipelining (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor(),
          pybind11::arg("stride"),
          pybind11::arg("dilation"),
          pybind11::arg("num_streams") = 4);
}
