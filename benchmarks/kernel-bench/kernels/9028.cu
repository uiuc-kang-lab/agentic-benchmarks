#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Basic 1D convolution kernel for a sub-batch
__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int B,                // Number of batches in this segment
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;
    int base = o * stride;
    int batch_offset = b * (in_channels * in_size);
    int weight_oc_offset = oc * (in_channels * kernel_size);
    for (int ic = 0; ic < in_channels; ++ic) {
        int in_offset = batch_offset + ic * in_size;
        int weight_offset = weight_oc_offset + ic * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            int pos = base + k * dilation;
            if (pos < in_size) {
                sum += x[in_offset + pos] * weight[weight_offset + k];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[oc];
    }
    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
}

// Forward function using CUDA streams to pipeline kernel execution and memory transfers
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation,
    int num_streams = 4  // default number of streams
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }
    
    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");
    
    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Determine the number of streams to use based on the batch size
    int streams_to_use = std::min(num_streams, B);
    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the batch dimension among the available streams
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
        conv1d_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_data + start_B * in_channels * in_size,
            weight_data,
            bias_data,
            output_data + start_B * out_channels * out_size,
            current_B,
            in_channels,
            in_size,
            out_channels,
            kernel_size,
            out_size,
            stride,
            dilation
        );
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < streams_to_use; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with pipelined streams (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor(),
          pybind11::arg("stride"),
          pybind11::arg("dilation"),
          pybind11::arg("num_streams") = 4);
}
