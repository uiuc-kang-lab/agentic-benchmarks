#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
float* output, const float* input, const float* weight,
const float* bias, int batch_size, int in_channels,
int out_channels, int in_height, int in_width,
int kernel_h, int kernel_w, int stride_h, int stride_w,
int pad_h, int pad_w, int out_height, int out_width) {

    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int h = blockIdx.z * blockDim.x + threadIdx.x;
    int w = threadIdx.y;
    
    if (h >= out_height || w >= out_width)
        return;
        
    float sum = bias ? bias[out_ch] : 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                if ((h + pad_h - kh) % stride_h == 0 &&
                    (w + pad_w - kw) % stride_w == 0) {
                    
                    int h_in = (h + pad_h - kh) / stride_h;
                    int w_in = (w + pad_w - kw) / stride_w;
                    
                    if (h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {
                        
                        int in_idx = ((batch_idx * in_channels + in_ch) *
                                     in_height + h_in) * in_width + w_in;
                        int weight_idx = ((out_ch * in_channels + in_ch) *
                                         kernel_h + kh) * kernel_w + kw;
                        
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    int out_idx = ((batch_idx * out_channels + out_ch) *
                   out_height + h) * out_width + w;
    output[out_idx] = sum;
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
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int out_height = (in_height - 1) * stride[0] -
                     2 * padding[0] + kernel_h + output_padding[0];
    int out_width = (in_width - 1) * stride[1] -
                    2 * padding[1] + kernel_w + output_padding[1];
    
    auto output = torch::zeros({batch_size, out_channels,
                               out_height, out_width},
                               input.options());
    
    dim3 threads(32, 32);
    dim3 blocks(batch_size, out_channels,
               (out_height + threads.x - 1) / threads.x);
    
    int shared_mem_size = threads.x * threads.y * sizeof(float);
    
    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_h, kernel_w,
        stride[0], stride[1], padding[0], padding[1],
        out_height, out_width);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}