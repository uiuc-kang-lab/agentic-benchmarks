#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Use warp-level parallelism (32 threads per warp)
    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    
    int total_elements = batch_size * out_channels * output_h * output_w;
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    
    // Each warp processes multiple output elements
    for (int idx = warp_idx; idx < total_elements; idx += gridDim.x * blockDim.x / warp_size) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int oc = (idx / (output_w * output_h)) % out_channels;
        int b = idx / (out_channels * output_h * output_w);
        
        int in_ch = oc / channels_per_group;
        int weight_ch = oc % channels_per_group;
        
        float partial_sum = 0.0f;
        
        // Distribute kernel elements across warp threads
        for (int k = lane_id; k < kernel_size * kernel_size; k += warp_size) {
            int kh = k / kernel_size;
            int kw = k % kernel_size;
            
            int h_in = h_out * stride + kh - padding;
            int w_in = w_out * stride + kw - padding;
            
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_idx = b * (in_channels * input_h * input_w)
                              + in_ch * (input_h * input_w)
                              + h_in * input_w
                              + w_in;
                              
                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                               + weight_ch * (kernel_size * kernel_size)
                               + kh * kernel_size
                               + kw;
                               
                partial_sum += input[input_idx] * weight[weight_idx];
            }
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        
        // First thread in warp writes the result
        if (lane_id == 0) {
            float final_sum = partial_sum;
            if (bias != nullptr) {
                final_sum += bias[oc];
            }
            
            output[b * out_channels * output_h * output_w +
                   oc * output_h * output_w +
                   h_out * output_w +
                   w_out] = final_sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;
    
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    // Adjust block size to be multiple of warp size
    int threads = 256;
    int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;
    
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}