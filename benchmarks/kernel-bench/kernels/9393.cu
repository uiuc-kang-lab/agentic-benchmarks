#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void stride_loop_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int oc = blockIdx.z;

    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    int w_out_base = bx * blockDim.x + tx;
    int h_out_base = by * blockDim.y + ty;

    const float bias_val = bias ? bias[oc] : 0.0f;

    for (int h_out = h_out_base; h_out < height_out; h_out += stride_y) {
        for (int w_out = w_out_base; w_out < width_out; w_out += stride_x) {
            if (w_out < width_out && h_out < height_out) {
                for (int b = 0; b < batch_size; ++b) {
                    float sum = bias_val;
                    const int h_in_start = h_out * stride - pad_h;
                    const int w_in_start = w_out * stride - pad_w;

                    #pragma unroll 4
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int in_batch_offset = b * in_channels * input_height * input_width;
                        const int in_channel_offset = ic * input_height * input_width;
                        const int weight_channel_offset = (oc * in_channels + ic) * kernel_h * kernel_w;

                        for (int kh = 0; kh < kernel_h; ++kh) {
                            const int h_in = h_in_start + kh * dilation_h;
                            
                            if (h_in >= 0 && h_in < input_height) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int w_in = w_in_start + kw * dilation_w;
                                    
                                    if (w_in >= 0 && w_in < input_width) {
                                        const int x_idx = in_batch_offset + in_channel_offset +
                                                        h_in * input_width + w_in;
                                        const int w_idx = weight_channel_offset +
                                                        kh * kernel_w + kw;
                                        sum += x[x_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                    }

                    const int out_idx = b * out_channels * height_out * width_out +
                                      oc * height_out * width_out +
                                      h_out * width_out + w_out;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    // Optimize thread block dimensions for better occupancy
    // Use 16x16 thread blocks for better 2D spatial locality
    dim3 threads(16, 16);
    
    // Calculate grid dimensions to exactly cover the output dimensions
    // This avoids unnecessary thread divergence in the kernel
    dim3 blocks(
        (width_out + threads.x - 1) / threads.x,
        (height_out + threads.y - 1) / threads.y,
        out_channels
    );

    stride_loop_conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        height_out,
        width_out,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA)");
}