#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_coalesced_kernel(
float* output, const float* input, const float* weight,
const float* bias, int batch_size, int in_channels,
int out_channels, int in_height, int in_width,
int kernel_height, int kernel_width,
int stride_h, int stride_w,
int padding_h, int padding_w,
int out_height, int out_width) {

    extern __shared__ float shared_weight[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / (out_channels * out_height * out_width);
    int remainder = idx % (out_channels * out_height * out_width);
    int oc = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int oh = remainder / out_width;
    int ow = remainder % out_width;

    if (batch >= batch_size || oc >= out_channels ||
        oh >= out_height || ow >= out_width) return;

    // Load weights into shared memory
    int tid = threadIdx.x;
    int weights_per_thread = (kernel_height * kernel_width * in_channels + blockDim.x - 1) / blockDim.x;
    for(int i = 0; i < weights_per_thread; i++) {
        int widx = tid + i * blockDim.x;
        if(widx < kernel_height * kernel_width * in_channels) {
            shared_weight[widx] = weight[oc * kernel_height * kernel_width * in_channels + widx];
        }
    }
    __syncthreads();

    float sum = bias ? bias[oc] : 0.0f;

    // Compute convolution with coalesced memory access
    for(int ic = 0; ic < in_channels; ic++) {
        for(int kh = 0; kh < kernel_height; kh++) {
            for(int kw = 0; kw < kernel_width; kw++) {
                int ih = (oh + padding_h - kh) / stride_h;
                int iw = (ow + padding_w - kw) / stride_w;
                
                if(ih >= 0 && ih < in_height && iw >= 0 && iw < in_width &&
                   (oh + padding_h - kh) % stride_h == 0 &&
                   (ow + padding_w - kw) % stride_w == 0) {
                    
                    int input_idx = ((batch * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw;
                    
                    sum += input[input_idx] * shared_weight[weight_idx];
                }
            }
        }
    }

    output[((batch * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    auto input = x.contiguous();
    auto weights = weight.contiguous();
    auto bias_opt = bias.has_value() ? bias.value().contiguous() : torch::Tensor();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int out_height = (in_height - 1) * stride[0] - 2 * padding[0] +
                     kernel_height + output_padding[0];
    int out_width = (in_width - 1) * stride[1] - 2 * padding[1] +
                    kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              input.options());

    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    const int shared_mem_size = kernel_height * kernel_width * in_channels * sizeof(float);

    conv_transpose2d_coalesced_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias_opt.defined() ? bias_opt.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_height, kernel_width,
        stride[0], stride[1],
        padding[0], padding[1],
        out_height, out_width);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}