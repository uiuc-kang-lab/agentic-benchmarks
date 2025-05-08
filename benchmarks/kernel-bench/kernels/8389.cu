#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses stride loops to cover the output spatial and batch/channel dimensions
// in case the workload is larger than the available number of threads in each dimension.
__global__ void conv_transpose2d_kernel_stride(
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

    // Determine strides for each dimension
    int stride_z = gridDim.z * blockDim.z;
    int index_z = blockIdx.z * blockDim.z + threadIdx.z;

    int stride_x = gridDim.x * blockDim.x;
    int start_x = blockIdx.x * blockDim.x + threadIdx.x;

    int stride_y = gridDim.y * blockDim.y;
    int start_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Loop over the combined batch and output channel dimension
    for (int linear_idx = index_z; linear_idx < batch_size * out_channels; linear_idx += stride_z) {
        int batch = linear_idx / out_channels;
        int out_ch = linear_idx % out_channels;
        
        // Loop over the output spatial dimensions with grid-stride looping
        for (int out_y = start_y; out_y < output_height; out_y += stride_y) {
            for (int out_x = start_x; out_x < output_width; out_x += stride_x) {
                float sum = 0.0f;
                
                // Loop over input channels and kernel dimensions
                for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            int in_x = out_x + pad_w - kw;
                            int in_y = out_y + pad_h - kh;
                            
                            // Check if the computed input index aligns with the stride
                            if ((in_x % stride_w) == 0 && (in_y % stride_h) == 0) {
                                in_x /= stride_w;
                                in_y /= stride_h;
                                
                                // Validate the input boundaries
                                if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                                    int input_idx = batch * in_channels * input_height * input_width +
                                                    in_ch * input_height * input_width +
                                                    in_y * input_width + in_x;
                                    
                                    int weight_idx = in_ch * out_channels * kernel_height * kernel_width +
                                                     out_ch * kernel_height * kernel_width +
                                                     kh * kernel_width + kw;
                                    
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }
                
                // Add bias if provided
                if (bias)
                    sum += bias[out_ch];
                
                int output_idx = batch * out_channels * output_height * output_width +
                                 out_ch * output_height * output_width +
                                 out_y * output_width + out_x;
                output[output_idx] = sum;
            }
        }
    }
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

    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    // Configure block and grid dimensions; using 3D grid-stride loops for all dimensions
    // Threads cover x and y spatial dimensions; z dimension covers (batch*out_channels)
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        (batch_size * out_channels + threads.z - 1) / threads.z
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv_transpose2d_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(),
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
    m.def("forward", &conv_transpose2d_cuda, "Optimized ConvTranspose2D with stride loops (CUDA)");
}
