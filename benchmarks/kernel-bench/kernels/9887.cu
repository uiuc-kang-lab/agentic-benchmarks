#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum constant memory capacity for weights (in number of float elements)
#define MAX_CONST_WEIGHT_SIZE 4096

// Declare constant memory for weight values
__constant__ float d_const_weight[MAX_CONST_WEIGHT_SIZE];

// Optimized depthwise convolution kernel using constant memory and grid-stride loops
__global__ void depthwise_conv2d_const_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    int channels_per_group,
    int weight_numel
) {
    int total_elements = batch_size * out_channels * output_h * output_w;
    
    // Use grid-stride loop for flexibility and potential higher occupancy
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Decode flattened index to (b, oc, h_out, w_out)
        int w_out = idx % output_w;
        int tmp = idx / output_w;
        int h_out = tmp % output_h;
        tmp /= output_h;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        int in_ch = oc / channels_per_group;
        int weight_ch = oc % channels_per_group;

        float sum = 0.0f;

        // Precompute starting indices for input based on stride and padding
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        // Loop over convolution kernel
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_in_start + kh;
            bool valid_h = (h_in >= 0 && h_in < input_h);
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_in_start + kw;
                if (valid_h && (w_in >= 0 && w_in < input_w)) {
                    int input_idx = b * (in_channels * input_h * input_w)
                                  + in_ch * (input_h * input_w)
                                  + h_in * input_w
                                  + w_in;
                    int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                                   + weight_ch * (kernel_size * kernel_size)
                                   + kh * kernel_size
                                   + kw;
                    // Access constant memory for weight
                    float w_val = (weight_idx < weight_numel) ? d_const_weight[weight_idx] : 0.0f;
                    sum += input[input_idx] * w_val;
                }
            }
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        int output_idx = b * (out_channels * output_h * output_w)
                        + oc * (output_h * output_w)
                        + h_out * output_w
                        + w_out;
        output[output_idx] = sum;
    }
}

// Helper function to load weight data into constant memory
void load_weights_to_constant_memory(const float* weight, int numel) {
    cudaMemcpyToSymbol(d_const_weight, weight, numel * sizeof(float));
}

// Forward function callable from Python
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
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONST_WEIGHT_SIZE, "Weight tensor exceeds constant memory limit");
    load_weights_to_constant_memory(weight.data_ptr<float>(), weight_numel);

    int total_elements = batch_size * out_channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_const_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
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
        channels_per_group,
        weight_numel
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution using Constant Memory",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
