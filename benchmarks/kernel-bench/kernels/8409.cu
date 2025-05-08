#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel assigns one output pixel to each thread, eliminating the need for atomic operations
// Each thread computes its own unique output value by iterating over input channels and kernel elements
__global__ void output_pixel_conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int output_height,
    int output_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_height * output_width;
    
    // Grid-stride loop to cover all output pixels
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        // Decode the 1D index into 4D indices: b, out_channel, out_height, out_width
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int oc = (idx / (output_width * output_height)) % out_channels;
        int b = idx / (output_width * output_height * out_channels);
        
        float sum = 0.0f;
        
        // Iterate over input channels
        for (int ic = 0; ic < in_channels; ic++) {
            // Precompute base addresses for input and weight for this channel
            int input_base = b * (in_channels * input_height * input_width) + ic * (input_height * input_width);
            int weight_base = ic * (out_channels * kernel_height * kernel_width) + oc * (kernel_height * kernel_width);
            
            // Loop over the kernel spatial dimensions
            for (int kh = 0; kh < kernel_height; kh++) {
                int in_y = h + pad_h - kh;
                // Only consider valid input positions that align with the stride
                if (in_y < 0 || in_y % stride_h != 0) continue;
                int i_h = in_y / stride_h;
                if (i_h < 0 || i_h >= input_height) continue;
                
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = w + pad_w - kw;
                    if (in_x < 0 || in_x % stride_w != 0) continue;
                    int i_w = in_x / stride_w;
                    if (i_w < 0 || i_w >= input_width) continue;
                    
                    float inp = input[input_base + i_h * input_width + i_w];
                    float wgt = weight[weight_base + kh * kernel_width + kw];
                    sum += inp * wgt;
                }
            }
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());
    
    int total = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    
    output_pixel_conv_transpose2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1]
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "Output-pixel based ConvTranspose2D forward (CUDA)");
}
