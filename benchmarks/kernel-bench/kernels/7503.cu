#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ConvTranspose2d using thread stride loops
__global__ void convTranspose2DStrideKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int H_out,
    int W_out,
    int groups,
    bool bias_present) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * H_out * W_out;
    int stride_val = blockDim.x * gridDim.x;

    for (int index = idx; index < total; index += stride_val) {
        // Map linear index to (b, out_ch, h, w)
        int w = index % W_out;
        int tmp = index / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int out_ch = tmp % out_channels;
        int b = tmp / out_channels;

        float out_val = 0.0f;

        // Determine which group the output channel belongs to
        int in_channels_per_group = in_channels / groups;
        int out_channels_per_group = out_channels / groups;
        int group = out_ch / out_channels_per_group;
        int out_ch_mod = out_ch % out_channels_per_group;

        // For each input channel in the corresponding group
        for (int in_ch = group * in_channels_per_group; in_ch < (group + 1) * in_channels_per_group; in_ch++) {
            for (int k_h = 0; k_h < kernel_size; k_h++) {
                int h_in = h + padding - k_h;
                if (h_in % stride != 0) continue;
                h_in /= stride;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int k_w = 0; k_w < kernel_size; k_w++) {
                    int w_in = w + padding - k_w;
                    if (w_in % stride != 0) continue;
                    w_in /= stride;
                    if (w_in < 0 || w_in >= W_in) continue;

                    int input_index = ((b * in_channels + in_ch) * H_in + h_in) * W_in + w_in;
                    int weight_index = ((in_ch * out_channels_per_group + out_ch_mod) * kernel_size + k_h) * kernel_size + k_w;
                    out_val += input[input_index] * weight[weight_index];
                }
            }
        }
        
        if (bias_present) {
            out_val += bias[out_ch];
        }

        int output_index = ((b * out_channels + out_ch) * H_out + h) * W_out + w;
        output[output_index] = out_val;
    }
}

// Host function for ConvTranspose2d forward pass with stride loops
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    bool bias_present = false;
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
        bias_present = true;
    }

    // Get dimensions
    int batch = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int kernel_size = weight.size(2); // square kernel assumed
    int out_channels = weight.size(1) * groups; // weight shape: [in_channels, out_channels/groups, kernel_size, kernel_size]
    
    // Compute output spatial dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Allocate output tensor
    auto output = torch::zeros({batch, out_channels, H_out, W_out}, x.options());

    int total_threads = batch * out_channels * H_out * W_out;
    int block_size = 128;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // Launch the CUDA kernel
    convTranspose2DStrideKernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias_tensor.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        H_in,
        W_in,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out,
        groups,
        bias_present
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with stride loops (CUDA)");
}
