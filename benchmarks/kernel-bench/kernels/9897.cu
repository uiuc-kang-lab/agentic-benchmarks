#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for depthwise convolution using __ldg() for read-only global memory accesses
__global__ void depthwise_conv2d_kernel(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride_param,
    int padding,
    int channels_per_group
) {
    int total_elements = batch_size * out_channels * output_h * output_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop over output elements
    for (int index = tid; index < total_elements; index += stride) {
        // Compute output indices: (b, oc, h_out, w_out)
        int w_out = index % output_w;
        int temp = index / output_w;
        int h_out = temp % output_h;
        temp /= output_h;
        int oc = temp % out_channels;
        int b  = temp / out_channels;

        // Determine input channel and corresponding weight channel for depthwise convolution
        int in_ch = oc / channels_per_group;
        // int weight_ch = oc % channels_per_group;   // Not used separately in this version, weight layout is adjusted below

        float sum = 0.0f;
        // Compute starting indices in the input
        int h_in_start = h_out * stride_param - padding;
        int w_in_start = w_out * stride_param - padding;

        // Compute effective kernel bounds to avoid conditional checks inside the inner loops
        int kh_start = (h_in_start < 0) ? -h_in_start : 0;
        int kh_end   = (h_in_start + kernel_size > input_h) ? (input_h - h_in_start) : kernel_size;
        int kw_start = (w_in_start < 0) ? -w_in_start : 0;
        int kw_end   = (w_in_start + kernel_size > input_w) ? (input_w - w_in_start) : kernel_size;

        // Loop over the kernel window using the effective bounds
        for (int kh = kh_start; kh < kh_end; ++kh) {
            int h_in = h_in_start + kh;
            for (int kw = kw_start; kw < kw_end; ++kw) {
                int w_in = w_in_start + kw;
                
                // Compute linear index for input pixel
                int input_idx = b * (in_channels * input_h * input_w) +
                                in_ch * (input_h * input_w) +
                                h_in * input_w +
                                w_in;

                // Compute linear index for weight. Weight is assumed to be laid out as:
                // [in_channels, channels_per_group, kernel_size, kernel_size]
                // This is equivalent to depthwise layout with out_channels = in_channels * channels_per_group
                int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size) +
                                 (oc % channels_per_group) * (kernel_size * kernel_size) +
                                 kh * kernel_size +
                                 kw;

                // Use __ldg to load from global memory (read-only) to help with 128-bit aligned loads
                float inp = __ldg(&input[input_idx]);
                float wei = __ldg(&weight[weight_idx]);
                sum += inp * wei;
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += __ldg(&bias[oc]);
        }

        // Write the result to the output tensor
        output[index] = sum;
    }
}

// C++ binding code
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride_param,
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

    int output_h = (input_h + 2 * padding - kernel_size) / stride_param + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride_param + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int total_elements = batch_size * out_channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

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
        stride_param,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
