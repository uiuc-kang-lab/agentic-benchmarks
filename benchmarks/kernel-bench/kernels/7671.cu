#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// CUDA kernel using warp-level primitives for reduction
__global__ void conv3d_warp_reduction_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Each warp computes one output element.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32;  // each warp has 32 threads
    int lane = global_thread_id % 32;

    // Total number of output elements
    int total_output = batch_size * out_channels * out_depth * out_height * out_width;
    if (warp_id >= total_output) return;

    // Decode warp_id into output coordinates (b, c_out, d_out, h_out, w_out)
    int temp = warp_id;
    int w_out = temp % out_width;
    temp /= out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int d_out = temp % out_depth;
    temp /= out_depth;
    int c_out = temp % out_channels;
    temp /= out_channels;
    int b = temp;  // remaining value

    // Determine which group and corresponding input channels
    int out_channels_per_group = out_channels / groups;  // assumed divisible
    int group = c_out / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    // Flatten the iteration over the kernel and input channels for this group
    int total_iter = in_channels_per_group * kernel_d * kernel_h * kernel_w;

    float partial_sum = 0.0f;

    // Each thread in the warp processes a subset of the iterations
    for (int i = lane; i < total_iter; i += 32) {
        int ic = i / (kernel_d * kernel_h * kernel_w);
        int rem = i % (kernel_d * kernel_h * kernel_w);
        int kd = rem / (kernel_h * kernel_w);
        int rem2 = rem % (kernel_h * kernel_w);
        int kh = rem2 / kernel_w;
        int kw = rem2 % kernel_w;

        int in_c = group * in_channels_per_group + ic;
        int d_in = d_out * stride - padding + kd * dilation;
        int h_in = h_out * stride - padding + kh * dilation;
        int w_in = w_out * stride - padding + kw * dilation;

        // Check input boundaries
        if (d_in >= 0 && d_in < in_depth &&
            h_in >= 0 && h_in < in_height &&
            w_in >= 0 && w_in < in_width) {
            // Compute linear index for input in NCDHW layout
            int input_idx = (((b * in_channels + in_c) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
            // Compute linear index for weight in OI(DHW) layout where O = out_channels,
            // I = in_channels_per_group
            int weight_idx = (((c_out * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // The first lane writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            partial_sum += bias[c_out];
        }
        int out_idx = (((b * out_channels + c_out) * out_depth + d_out) * out_height + h_out) * out_width + w_out;
        output[out_idx] = partial_sum;
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
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Get weight dimensions
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Prepare output tensor
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Total number of output elements
    int total_output = batch_size * out_channels * out_depth * out_height * out_width;
    // Each warp computes one output element, so total threads = total_output * 32
    int threads_per_block = 256;
    int total_warps = total_output;
    int total_threads = total_warps * 32;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv3d_warp_reduction_kernel<<<num_blocks, threads_per_block>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with warp-level reduction (CUDA)");
}
