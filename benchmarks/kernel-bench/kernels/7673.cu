#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handles.h>
#include <ATen/cudnn/Descriptors.h>
#include <cudnn.h>

#define BLOCK_SIZE 256
#define SMALL_INPUT_THRESHOLD 32  // Threshold to switch between custom and cuDNN

cudnnDataType_t getCudnnDataType(at::ScalarType type) {
    switch (type) {
        case at::ScalarType::Float: return CUDNN_DATA_FLOAT;
        case at::ScalarType::Double: return CUDNN_DATA_DOUBLE;
        case at::ScalarType::Half: return CUDNN_DATA_HALF;
        default: TORCH_CHECK(false, "Unsupported data type for cuDNN");
    }
}

__global__ void conv3d_strided_kernel(
    float* output, const float* input, const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int kernel_d, int kernel_h, int kernel_w,
    int out_depth, int out_height, int out_width,
    int stride, int padding, int dilation, int groups) {
    
    // Custom kernel implementation for small inputs
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int stride_size = gridDim.x * blockDim.x;
    
    for(int idx = tid; idx < total_elements; idx += stride_size) {
        // Calculate output position
        const int w_out = idx % out_width;
        const int h_out = (idx / out_width) % out_height;
        const int d_out = (idx / (out_width * out_height)) % out_depth;
        const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
        const int b = idx / (out_width * out_height * out_depth * out_channels);
        
        float sum = 0.0f;
        
        const int group = c_out / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;
        
        for(int ic = 0; ic < in_channels_per_group; ic++) {
            const int in_channel = group * in_channels_per_group + ic;
            
            #pragma unroll
            for(int kd = 0; kd < kernel_d; kd++) {
                const int d_in = d_out * stride - padding + kd * dilation;
                if(d_in < 0 || d_in >= in_depth) continue;
                
                #pragma unroll
                for(int kh = 0; kh < kernel_h; kh++) {
                    const int h_in = h_out * stride - padding + kh * dilation;
                    if(h_in < 0 || h_in >= in_height) continue;
                    
                    #pragma unroll
                    for(int kw = 0; kw < kernel_w; kw++) {
                        const int w_in = w_out * stride - padding + kw * dilation;
                        if(w_in < 0 || w_in >= in_width) continue;
                        
                        sum += input[((b * in_channels + in_channel) * in_depth + d_in) * 
                                    in_height * in_width + h_in * in_width + w_in] *
                               weight[((c_out * in_channels_per_group + ic) * kernel_d + kd) * 
                                     kernel_h * kernel_w + kh * kernel_w + kw];
                    }
                }
            }
        }
        
        if(bias != nullptr) {
            sum += bias[c_out];
        }
        
        output[idx] = sum;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    
    bool use_custom = (in_depth <= SMALL_INPUT_THRESHOLD && 
                      in_height <= SMALL_INPUT_THRESHOLD && 
                      in_width <= SMALL_INPUT_THRESHOLD);
    
    if (use_custom) {
        auto out_channels = weight.size(0);
        auto kernel_d = weight.size(2);
        auto kernel_h = weight.size(3);
        auto kernel_w = weight.size(4);
        
        auto out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
        auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
        auto out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
        
        auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
        
        const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        conv3d_strided_kernel<<<num_blocks, BLOCK_SIZE>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            batch_size, in_channels, out_channels,
            in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            out_depth, out_height, out_width,
            stride, padding, dilation, groups
        );
        
        return output;
    } else {
        // Use cuDNN path for larger inputs
        return at::cudnn_convolution(
            input, weight, bias,
            {padding, padding, padding},
            {stride, stride, stride},
            {dilation, dilation, dilation},
            groups, false
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid 3D convolution forward (CUDA)");
}