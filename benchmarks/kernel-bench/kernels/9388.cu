#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel that computes convolution for a single batch element using provided batch index 'b'.
__global__ void streamed_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int b,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int oc    = blockIdx.z;

    if (w_out >= out_width || h_out >= out_height || oc >= out_channels) return;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Loop over input channels and kernel dimensions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + kh * dilation_h - pad_h;
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int x_idx = b * in_channels * input_height * input_width + 
                                ic * (input_height * input_width) +
                                h_in * input_width + w_in;
                    int w_idx = oc * in_channels * kernel_h * kernel_w + 
                                ic * kernel_h * kernel_w +
                                kh * kernel_w + kw;
                    sum += x[x_idx] * weight[w_idx];
                }
            }
        }
    }

    int out_idx = b * out_channels * out_height * out_width + 
                  oc * (out_height * out_width) + 
                  h_out * out_width + w_out;
    output[out_idx] = sum;
}

// Forward function using multiple CUDA streams to overlap kernel execution and memory transfers

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size   = x.size(0);
    int in_channels  = x.size(1);
    int input_height = x.size(2);
    int input_width  = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h     = weight.size(2);
    int kernel_w     = weight.size(3);

    int pad_h      = std::get<0>(padding);
    int pad_w      = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int out_height = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int out_width  = (input_width  + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Use multiple streams to overlap computation with any asynchronous memory transfers
    // Choose the number of streams (using up to 4 streams or batch_size if smaller)
    int num_streams = batch_size < 4 ? batch_size : 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Setup kernel launch dimensions for a single batch element
    dim3 threads(16, 16);
    dim3 grid((out_width + threads.x - 1) / threads.x,
              (out_height + threads.y - 1) / threads.y,
              out_channels);

    // Launch a kernel for each batch element asynchronously on a separate stream
    for (int b = 0; b < batch_size; b++) {
        int stream_id = b % num_streams;
        streamed_conv2d_kernel<<<grid, threads, 0, streams[stream_id]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            b,
            in_channels,
            input_height,
            input_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA) with streams overlapping computation and memory transfers");
}
