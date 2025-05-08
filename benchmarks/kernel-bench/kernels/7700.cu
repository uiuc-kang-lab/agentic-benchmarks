#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Target block size; tuning might be needed per hardware
#define BLOCK_SIZE 1024

// Optimized 3D convolution kernel with manual loop unrolling
__global__ void conv3d_optimized_unroll_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each block processes one output channel for one batch sample
    const int oc = blockIdx.x; // output channel
    const int batch_id = blockIdx.y; // batch index

    // Total number of output elements (spatial volume)
    const int total_output = out_depth * out_height * out_width;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Load bias for this output channel
    float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;

    // Iterate through output elements assigned to this thread
    for (int out_idx = tid; out_idx < total_output; out_idx += block_size) {
        // Map linear index to 3D coordinates
        int od = out_idx / (out_height * out_width);
        int rem = out_idx % (out_height * out_width);
        int oh = rem / out_width;
        int ow = rem % out_width;

        float sum = 0.0f;

        // Precompute starting indices in the input volume for the convolution window
        int id_start = od * stride - padding;
        int ih_start = oh * stride - padding;
        int iw_start = ow * stride - padding;

        // Loop over input channels; if in_channels is small, this loop may be unrolled
        #pragma unroll
        for (int ic = 0; ic < in_channels; ic++) {
            // Manually unroll kernel depth loop
            #pragma unroll
            for (int kd = 0; kd < kernel_d; kd++) {
                int id = id_start + kd * dilation;
                if (id < 0 || id >= in_depth) continue;

                #pragma unroll
                for (int kh = 0; kh < kernel_h; kh++) {
                    int ih = ih_start + kh * dilation;
                    if (ih < 0 || ih >= in_height) continue;

                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int iw = iw_start + kw * dilation;
                        if (iw < 0 || iw >= in_width) continue;

                        // Compute linear indices for input and weight
                        int input_idx = (((batch_id * in_channels + ic) * in_depth + id) * in_height + ih) * in_width + iw;
                        int weight_idx = (((oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Write the computed output adding the bias
        int output_idx = (((batch_id * out_channels + oc) * out_depth + od) * out_height + oh) * out_width + ow;
        output[output_idx] = sum + bias_val;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 is supported");
    auto bias = bias_opt.value_or(at::Tensor());

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Configure grid: each block handles one output channel and one batch sample
    dim3 grid(out_channels, batch_size);
    int threads = BLOCK_SIZE;

    conv3d_optimized_unroll_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
        out_height,
        out_width,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward with manual loop unrolling (CUDA)");
}
