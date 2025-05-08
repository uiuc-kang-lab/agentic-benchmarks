#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// The kernel uses __ldg() for read-only loads and assumes that the underlying memory is
// 128-bit aligned for efficient global memory operations.

// Split the convolution into multiple streams for concurrent execution
#define NUM_STREAMS 4

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    int stream_id,
    int streams_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_channels * out_size;
    
    // Distribute work across streams
    int items_per_stream = (total + streams_total - 1) / streams_total;
    int stream_start = stream_id * items_per_stream;
    int stream_end = min(stream_start + items_per_stream, total);
    
    idx += stream_start;
    if (idx >= stream_end) return;

    // Compute output indices
    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    float sum = 0.0f;
    int base_input = o * stride;

    // Loop over input channels and kernel width
    for (int ic = 0; ic < in_channels; ++ic) {
        int x_offset = b * (in_channels * in_size) + ic * in_size;
        int w_offset = oc * (in_channels * kernel_size) + ic * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = base_input + k * dilation;
            if (input_pos < in_size) {
                // Use __ldg() for optimized read-only loads
                float x_val = __ldg(x + x_offset + input_pos);
                float w_val = __ldg(weight + w_offset + k);
                sum += x_val * w_val;
            }
        }
    }
    
    if (bias) {
        sum += __ldg(bias + oc);
    }

    output[out_idx] = sum;
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
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

    const int B = x.size(0);
    const int in_channels = x.size(1);
    const int in_size = x.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");
    
    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const int total_elements = B * out_channels * out_size;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    conv1d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, in_channels, in_size,
        out_channels, kernel_size, out_size,
        stride, dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with __ldg optimizations");
}
