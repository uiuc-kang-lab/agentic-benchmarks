#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Scatter-based depthwise convolution kernel using atomicAdd for necessary race conditions.
// Each thread processes one input element and scatters its contribution to all output positions
// for which it is in the receptive field. Atomic operations are used only at the final update
// to global memory to safely accumulate contributions from multiple threads.

__global__ void scatter_atomic_depthwise_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int channels_per_group,
    int kernel_size,
    int stride,
    int padding,
    int output_h,
    int output_w
) {
    // Each thread processes one input element
    int total_elements = batch_size * in_channels * input_h * input_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    // Decompose linear index into (b, in_ch, h, w)
    int w = index % input_w;
    int tmp = index / input_w;
    int h = tmp % input_h;
    tmp = tmp / input_h;
    int in_ch = tmp % in_channels;
    int b = tmp / in_channels;

    float input_val = input[index];

    // For each kernel offset, determine if the input element contributes to a valid output location.
    for (int kh = 0; kh < kernel_size; kh++) {
        int h_out_calc = h + padding - kh;
        if (h_out_calc < 0) continue;
        if (h_out_calc % stride != 0) continue;
        int h_out = h_out_calc / stride;
        if (h_out < 0 || h_out >= output_h) continue;

        for (int kw = 0; kw < kernel_size; kw++) {
            int w_out_calc = w + padding - kw;
            if (w_out_calc < 0) continue;
            if (w_out_calc % stride != 0) continue;
            int w_out = w_out_calc / stride;
            if (w_out < 0 || w_out >= output_w) continue;

            // For depthwise convolution, each input channel maps to multiple output channels
            // (one for each filter channel in the group):
            for (int r = 0; r < channels_per_group; r++) {
                int oc = in_ch * channels_per_group + r;
                // Weight tensor is assumed to be in shape: [in_channels, channels_per_group, kernel_size, kernel_size]
                int weight_index = in_ch * (channels_per_group * kernel_size * kernel_size)
                                  + r * (kernel_size * kernel_size)
                                  + kh * kernel_size + kw;
                float prod = input_val * weight[weight_index];

                // Compute output index: output is [batch_size, out_channels, output_h, output_w]
                int out_index = b * (in_channels * channels_per_group * output_h * output_w)
                              + oc * (output_h * output_w)
                              + h_out * output_w + w_out;

                // Atomic update to handle race conditions from multiple input contributions
                atomicAdd(&output[out_index], prod);
            }
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
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;

    // Calculate output dimensions
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    // Initialize output tensor. We initialize with zeros and add bias later if provided.
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    // Launch one thread per input element
    int total_elements = batch_size * in_channels * input_h * input_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    scatter_atomic_depthwise_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        channels_per_group,
        kernel_size,
        stride,
        padding,
        output_h,
        output_w
    );

    // If bias is provided, add it to the output (broadcasting over batch and spatial dimensions)
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
        output = output + bias.value().view({1, out_channels, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Scatter Atomic Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
