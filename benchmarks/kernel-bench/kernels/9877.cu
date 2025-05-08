#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel assigns one warp per output pixel. Each warp's threads cooperatively compute the convolution sum
// by distributing the kernel window multiplications among warp lanes and then using warp-level reduction (__shfl_down_sync) to combine the partial sums.

__global__ void depthwise_conv2d_warp_reduced_kernel(
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
    // Each warp computes one output element
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    int total_outputs = batch_size * out_channels * output_h * output_w;
    if (warp_id >= total_outputs) return;

    // Decode the flat warp_id into output tensor coordinates (b, oc, h_out, w_out)
    int tmp = warp_id;
    int w_out = tmp % output_w;
    tmp /= output_w;
    int h_out = tmp % output_h;
    tmp /= output_h;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;
    int kernel_area = kernel_size * kernel_size;

    // Compute the top-left corner of the receptive field in the input
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    float sum = 0.0f;

    // Distribute the kernel window iterations among the warp lanes
    for (int i = lane; i < kernel_area; i += 32) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int h_in = h_in_start + kh;
        int w_in = w_in_start + kw;
        
        // Check validity of the input coordinates
        if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) sum += input[input_idx] * weight[weight_idx]; else sum += 0; // Avoid branch divergence
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            h_in * input_w +
                            w_in;
            int weight_idx = in_ch * (channels_per_group * kernel_area) +
                             weight_ch * kernel_area +
                             i;
            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // The first lane in the warp writes the accumulated value to the output
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = b * (out_channels * output_h * output_w) +
                         oc * (output_h * output_w) +
                         h_out * output_w +
                         w_out;
        output[output_idx] = sum;
    }
}

// The forward function prepares the launch configuration and calls the kernel

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

    int total_outputs = batch_size * out_channels * output_h * output_w;
    // Each warp (32 threads) computes one output element
    int warps_per_block = 256 / 32;  // using 256 threads per block
    int num_blocks = (total_outputs + warps_per_block - 1) / warps_per_block;

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_warp_reduced_kernel<<<num_blocks, 256>>>(
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
    m.def("forward", &forward, "Depthwise 2D Convolution with Warp Reduction (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
