#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ConvTranspose2d using shared memory
__global__ void convTranspose2DSharedKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_size,
    int stride,
    int padding,
    int groups,
    bool bias_present
) {
    extern __shared__ float shared_weight[];
    int tid = threadIdx.x;
    
    // Load weights into shared memory
    for (int i = tid; i < in_channels * out_channels * kernel_size * kernel_size; i += blockDim.x) {
        shared_weight[i] = weight[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    int total = batch * out_channels * height_out * width_out;
    if (idx >= total)
        return;

    // Map linear index to (b, out_ch, h, w)
    int w = idx % width_out;
    int tmp = idx / width_out;
    int h = tmp % height_out;
    tmp /= height_out;
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
            if (h_in < 0 || h_in >= height_in) continue;

            for (int k_w = 0; k_w < kernel_size; k_w++) {
                int w_in = w + padding - k_w;
                if (w_in % stride != 0) continue;
                w_in /= stride;
                if (w_in < 0 || w_in >= width_in) continue;

                int input_index = ((b * in_channels + in_ch) * height_in + h_in) * width_in + w_in;
                int weight_index = (((in_ch % in_channels_per_group) * out_channels_per_group + out_ch_mod) * kernel_size + k_h) * kernel_size + k_w;
                out_val += input[input_index] * shared_weight[weight_index];
            }
        }
    }
    
    if (bias_present) {
        out_val += bias[out_ch];
    }

    int output_index = ((b * out_channels + out_ch) * height_out + h) * width_out + w;
    output[output_index] = out_val;
}

// Host function for ConvTranspose2d forward pass with shared memory
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
    int height_in = x.size(2);
    int width_in = x.size(3);
    int kernel_size = weight.size(2); // square kernel assumed
    int out_channels = weight.size(1) * groups; // weight shape: [in_channels, out_channels/groups, kernel_size, kernel_size]
    
    // Compute output spatial dimensions
    int height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Allocate output tensor
    auto output = torch::zeros({batch, out_channels, height_out, width_out}, x.options());

    int total_threads = batch * out_channels * height_out * width_out;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // Launch the CUDA kernel
    convTranspose2DSharedKernel<<<grid_size, block_size, in_channels * out_channels * kernel_size * kernel_size * sizeof(float), cudaStreamDefault>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_present ? bias_tensor.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        groups,
        bias_present
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with shared memory (CUDA)");
}
