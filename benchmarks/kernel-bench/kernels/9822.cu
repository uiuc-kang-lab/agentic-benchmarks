#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses warp-level primitives to reduce the need for shared memory in the convolution reduction.
// Each warp processes one output element by having each lane accumulate a partial sum over a subset of kernel elements,
// and then reduces the sum using __shfl_down_sync().

__global__ void depthwise_conv2d_warp_kernel(
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
    // Each warp processes one output element
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    int total_output = batch_size * out_channels * output_h * output_w;
    if (warpId >= total_output) return;
    
    // Decode warpId into output indices
    int tmp = warpId;
    int w_out = tmp % output_w;
    tmp /= output_w;
    int h_out = tmp % output_h;
    tmp /= output_h;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;
    
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    float sum = 0.0f;
    int kernel_total = kernel_size * kernel_size;
    
    // Each lane processes a subset of the kernel elements
    for (int idx = lane; idx < kernel_total; idx += warpSize) {
        int kh = idx / kernel_size;
        int kw = idx % kernel_size;
        int h_in = h_out * stride + kh - padding;
        int w_in = w_out * stride + kw - padding;
        
        if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            h_in * input_w + w_in;
            int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size) +
                             weight_ch * (kernel_size * kernel_size) +
                             kh * kernel_size + kw;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    // Warp-level reduction using __shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Lane 0 writes the computed output
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[b * (out_channels * output_h * output_w) +
               oc * (output_h * output_w) +
               h_out * output_w +
               w_out] = sum;
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

    int total_output = batch_size * out_channels * output_h * output_w;
    // Launch one warp per output element
    int total_threads = total_output * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_warp_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Warp-level Reduced Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
