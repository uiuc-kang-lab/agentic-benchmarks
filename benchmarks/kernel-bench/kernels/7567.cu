#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// Custom CUDA kernel for transposed 3D convolution with balanced workload distribution
__global__ void conv_transposed_3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int D_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int D_out,
    const int H_out,
    const int W_out,
    const int K_d,
    const int K_h,
    const int K_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int opad_d,
    const int opad_h,
    const int opad_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group
) {
    // Each thread computes one output element using a grid-stride loop
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;

    while (output_index < total) {
        int tmp = output_index;
        int w = tmp % W_out; tmp /= W_out;
        int h = tmp % H_out; tmp /= H_out;
        int d = tmp % D_out; tmp /= D_out;
        int c = tmp % C_out; tmp /= C_out;
        int n = tmp;

        float sum = 0.0f;
        // Determine group and relative output channel index
        int g = c / out_channels_per_group;  // group index
        int r = c % out_channels_per_group;  // relative output channel index

        // Pre-calculate base indices and offsets
        const int n_offset = n * C_in;
        const int d_base = d + pad_d;
        const int h_base = h + pad_h;
        const int w_base = w + pad_w;

        // Loop over the input channels in the corresponding group
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            int input_channel = g * in_channels_per_group + ic;
            // Pre-calculate input channel offset
            const int in_ch_offset = (n_offset + input_channel) * D_in;
            const int weight_ch_offset = input_channel * out_channels_per_group + r;

            // Loop over kernel depth
            for (int kd = 0; kd < K_d; kd++) {
                int d_in_temp = d_base - kd;
                if (d_in_temp < 0 || (d_in_temp % stride_d) != 0) continue;
                int d_in = d_in_temp / stride_d;
                if (d_in >= D_in) continue;

                const int d_offset = in_ch_offset + d_in * H_in;
                const int w_d_offset = weight_ch_offset * K_d + kd;

                // Loop over kernel height
                for (int kh = 0; kh < K_h; kh++) {
                    int h_in_temp = h_base - kh;
                    if (h_in_temp < 0 || (h_in_temp % stride_h) != 0) continue;
                    int h_in = h_in_temp / stride_h;
                    if (h_in >= H_in) continue;

                    const int h_offset = (d_offset + h_in) * W_in;
                    const int w_h_offset = (w_d_offset * K_h + kh) * K_w;

                    // Loop over kernel width
                    for (int kw = 0; kw < K_w; kw++) {
                        int w_in_temp = w_base - kw;
                        if (w_in_temp < 0 || (w_in_temp % stride_w) != 0) continue;
                        int w_in = w_in_temp / stride_w;
                        if (w_in >= W_in) continue;

                        // Use pre-calculated offsets for final index computation
                        int input_idx = h_offset + w_in;
                        int weight_idx = w_h_offset + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        // Write the result to the output tensor
        int out_idx = (((n * C_out + c) * D_out + d) * H_out + h) * W_out + w;
        output[out_idx] = sum;

        output_index += blockDim.x * gridDim.x;
    }
}


// Forward wrapper function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // x: [N, C_in, D_in, H_in, W_in]
    // weight: [C_in, out_channels_per_group, K_d, K_h, K_w]
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int D_in = x.size(2);
    const int H_in = x.size(3);
    const int W_in = x.size(4);

    const int out_channels_per_group = weight.size(1);
    const int C_out = groups * out_channels_per_group;

    const int K_d = weight.size(2);
    const int K_h = weight.size(3);
    const int K_w = weight.size(4);

    const int stride_d = stride[0];
    const int stride_h = stride[1];
    const int stride_w = stride[2];

    const int pad_d = padding[0];
    const int pad_h = padding[1];
    const int pad_w = padding[2];

    const int opad_d = output_padding[0];
    const int opad_h = output_padding[1];
    const int opad_w = output_padding[2];

    // Compute output dimensions
    const int D_out = (D_in - 1) * stride_d - 2 * pad_d + K_d + opad_d;
    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + opad_h;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + opad_w;

    // Create an output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, x.options());

    const int in_channels_per_group = C_in / groups;

    // Determine total number of output elements
    const int total_threads = N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    // Launch the kernel
    conv_transposed_3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        K_d, K_h, K_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        opad_d, opad_h, opad_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );
    cudaDeviceSynchronize();
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
