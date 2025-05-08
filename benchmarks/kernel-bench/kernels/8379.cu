#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to compute input indices
__device__ __forceinline__ bool compute_input_idx(
    int out_x, int out_y, int kw, int kh,
    int stride_w, int stride_h, int pad_w, int pad_h,
    int input_width, int input_height,
    int& in_x, int& in_y) {
    
    in_x = out_x + pad_w - kw;
    in_y = out_y + pad_h - kh;
    
    if (in_x % stride_w != 0 || in_y % stride_h != 0)
        return false;
        
    in_x /= stride_w;
    in_y /= stride_h;
    
    return (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height);
}

// Device function to compute memory offsets
__device__ __forceinline__ int compute_input_offset(
    int batch, int in_ch, int in_y, int in_x,
    int in_channels, int input_height, int input_width) {
    return batch * (in_channels * input_height * input_width) +
           in_ch * (input_height * input_width) +
           in_y * input_width + in_x;
}

__device__ __forceinline__ int compute_weight_offset(
    int in_ch, int out_ch, int kh, int kw,
    int out_channels, int kernel_height, int kernel_width) {
    return in_ch * (out_channels * kernel_height * kernel_width) +
           out_ch * (kernel_height * kernel_width) +
           kh * kernel_width + kw;
}

__device__ __forceinline__ int compute_output_offset(
    int batch, int out_ch, int out_y, int out_x,
    int out_channels, int output_height, int output_width) {
    return batch * (out_channels * output_height * output_width) +
           out_ch * (output_height * output_width) +
           out_y * output_width + out_x;
}

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
    
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_out_ch = blockIdx.z;
    const int batch = batch_out_ch / out_channels;
    const int out_ch = batch_out_ch % out_channels;
    
    if (out_x >= output_width || out_y >= output_height || batch >= batch_size)
        return;
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        #pragma unroll 2
        for (int kh = 0; kh < kernel_height; ++kh) {
            #pragma unroll 2
            for (int kw = 0; kw < kernel_width; ++kw) {
                int in_x, in_y;
                if (compute_input_idx(out_x, out_y, kw, kh, stride_w, stride_h,
                                    pad_w, pad_h, input_width, input_height,
                                    in_x, in_y)) {
                    
                    const int in_idx = compute_input_offset(batch, in_ch, in_y, in_x,
                                                          in_channels, input_height, input_width);
                    const int w_idx = compute_weight_offset(in_ch, out_ch, kh, kw,
                                                          out_channels, kernel_height, kernel_width);
                    
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    const int out_idx = compute_output_offset(batch, out_ch, out_y, out_x,
                                            out_channels, output_height, output_width);
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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    
    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] +
                             kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] +
                            kernel_width + output_padding[1];
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
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
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward (CUDA)");
}