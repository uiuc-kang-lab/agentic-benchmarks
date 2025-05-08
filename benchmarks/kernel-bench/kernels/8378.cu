#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define maximum constant memory size in number of floats (64KB / 4 = 16384 floats)
#define MAX_WEIGHT_ELEMENTS 16384

// Declare constant memory for the weight tensor
__constant__ float d_const_weight[MAX_WEIGHT_ELEMENTS];

// Kernel using constant memory for weight lookup
__global__ void conv_transpose2d_kernel_const(
    const float* __restrict__ input,
    const float* __restrict__ bias,
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

    // Map thread indices: x,y for spatial position; grid z for (batch, out_channel)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_idx = blockIdx.z; // linear index combining batch and out_channel
    int batch = linear_idx / out_channels;
    int out_ch = linear_idx % out_channels;

    if (out_x < output_width && out_y < output_height && batch < batch_size) {
        float sum = 0.0f;
        
        // Iterate over each input channel and kernel position
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int in_x = out_x + pad_w - kw;
                    int in_y = out_y + pad_h - kh;
                    
                    // Check alignment with stride
                    if (in_x % stride_w == 0 && in_y % stride_h == 0) {
                        in_x /= stride_w;
                        in_y /= stride_h;
                        
                        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                            float input_val = input[ batch * in_channels * input_height * input_width +
                                                      in_ch * input_height * input_width +
                                                      in_y * input_width + in_x];

                            int weight_index = in_ch * out_channels * kernel_height * kernel_width +
                                               out_ch * kernel_height * kernel_width +
                                               kh * kernel_width + kw;
                            float weight_val = d_const_weight[weight_index];

                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[out_ch];
        }

        // Write the computed output value
        int out_idx = batch * out_channels * output_height * output_width +
                      out_ch * output_height * output_width +
                      out_y * output_width + out_x;
        output[out_idx] = sum;
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

    // Get input dimensions
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    
    // Weight dimensions: weight shape [in_channels, out_channels, kernel_height, kernel_width]
    const int out_channels = weight.size(1);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0];
    const int output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    // Compute number of elements in weight and ensure it fits within constant memory
    size_t weight_numel = weight.numel();
    size_t weight_bytes = weight_numel * sizeof(float);
    TORCH_CHECK(weight_numel <= MAX_WEIGHT_ELEMENTS,
                "Weight tensor is too large for constant memory. Maximum allowed elements is ", MAX_WEIGHT_ELEMENTS);

    // Copy weight data into constant memory
    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight_bytes, 0, cudaMemcpyDeviceToDevice);

    // Configure kernel launch parameters
    dim3 threads(16, 16, 1);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size * out_channels  // each z-dimension thread covers one (batch, out_channel) pair
    );

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Launch the kernel
    conv_transpose2d_kernel_const<<<blocks, threads>>>(
        x.data_ptr<float>(),
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
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward with constant memory (CUDA)");
}
