#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_streamed_kernel(
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
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;
    int start_pos = o * stride;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* x_base = x + b * in_channels * in_size + ic * in_size;
        const float* w_base = weight + oc * in_channels * kernel_size + ic * kernel_size;
        
        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start_pos + k * dilation;
            if (pos < in_size && pos >= 0) {
                sum += x_base[pos] * w_base[k];
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];
    output[b * (out_channels * out_size) + oc * out_size + o] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int B = x.size(0);
    const int in_channels = x.size(1);
    const int in_size = x.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    // Stream configuration for H100
    constexpr int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Split batch dimension across streams
    const int batches_per_stream = (B + num_streams - 1) / num_streams;
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    for (int i = 0; i < num_streams; ++i) {
        const int b_start = i * batches_per_stream;
        const int b_end = std::min(b_start + batches_per_stream, B);
        const int current_B = b_end - b_start;
        if (current_B <= 0) break;

        const int elements = current_B * out_channels * out_size;
        const int threads = 256;
        const int blocks = (elements + threads - 1) / threads;

        conv1d_streamed_kernel<<<blocks, threads, 0, streams[i]>>>(
            x_data + b_start * in_channels * in_size,
            weight_data,
            bias_data,
            output_data + b_start * out_channels * out_size,
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

    // Synchronize and cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-parallel 1D convolution");
}