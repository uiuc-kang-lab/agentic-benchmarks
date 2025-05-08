#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// CUDA kernel with branchless conditional logic to minimize warp divergence
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int total = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = blockDim.x * gridDim.x;

    while (idx < total) {
        // Decode flattened index into (n, c_out, h_out, w_out)
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp = tmp / H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        // Compute channel grouping info
        int out_channels_per_group = C_out / groups;
        int in_channels_per_group = C_in / groups;
        int group_idx = c_out / out_channels_per_group;
        int input_channel_start = group_idx * in_channels_per_group;
        int input_channel_end = input_channel_start + in_channels_per_group;

        // Use bias if provided
        float value = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Loop over the kernel window
        for (int k_y = 0; k_y < kernel_h; ++k_y) {
            for (int k_x = 0; k_x < kernel_w; ++k_x) {
                // Compute potential input coordinates
                int h_in_possible = h_out + padding_h - k_y * dilation_h;
                int w_in_possible = w_out + padding_w - k_x * dilation_w;

                // Instead of branching on validity, compute a binary mask
                int cond1 = ((h_in_possible % stride_h) == 0) ? 1 : 0;
                int cond2 = ((w_in_possible % stride_w) == 0) ? 1 : 0;
                int h_in = h_in_possible / stride_h;
                int w_in = w_in_possible / stride_w;
                int cond3 = (h_in >= 0 && h_in < H_in) ? 1 : 0;
                int cond4 = (w_in >= 0 && w_in < W_in) ? 1 : 0;
                int valid = cond1 & cond2 & cond3 & cond4;

                // Accumulate contributions from the appropriate input channels
                for (int c_in = input_channel_start; c_in < input_channel_end; ++c_in) {
                    int input_index = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_index = (((c_in * out_channels_per_group) + (c_out - group_idx * out_channels_per_group)) 
                                          * kernel_h + k_y) * kernel_w + k_x;
                    value += valid * input[input_index] * weight[weight_index];
                }
            }
        }

        int output_index = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[output_index] = value;
        idx += stride_idx;
    }
}

// Wrapper function to configure and launch the kernel
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int stride_h = stride[0];
    int stride_w = stride[1];
    int padding_h = padding[0];
    int padding_w = padding[1];
    int output_padding_h = output_padding[0];
    int output_padding_w = output_padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int out_channels_per_group = weight.size(1);
    int C_out = out_channels_per_group * groups;

    // Compute output dimensions based on the transposed convolution formula
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int threads = 256;
    int total_elements = N * C_out * H_out * W_out;
    int blocks = (total_elements + threads - 1) / threads;

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D branchless forward (CUDA)");
}
