#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements a gather-based transposed convolution.
// Each thread computes one output pixel by iterating over input channels and kernel positions.
// By computing each output element exclusively, we avoid any race conditions and thus do not need atomic operations.
// This meets the recommendation to use atomic operations only where necessary (in this case, not at all).

__global__ void conv_transpose2d_gather_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // Can be nullptr if no bias
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int stride, int padding, int groups) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode the flat index into (n, oc, out_y, out_x)
    int out_x = index % W_out;
    int temp = index / W_out;
    int out_y = temp % H_out;
    temp = temp / H_out;
    int oc = temp % C_out;
    int n = temp / C_out;

    float sum = 0.f;

    // For groups == 1, loop over all input channels
    if (groups == 1) {
        for (int ic = 0; ic < C_in; ic++) {
            // Loop over kernel rows and columns
            for (int ky = 0; ky < K; ky++) {
                int in_y = out_y + padding - ky;
                if (in_y % stride != 0) continue;
                in_y /= stride;
                if (in_y < 0 || in_y >= H_in) continue;
                
                for (int kx = 0; kx < K; kx++) {
                    int in_x = out_x + padding - kx;
                    if (in_x % stride != 0) continue;
                    in_x /= stride;
                    if (in_x < 0 || in_x >= W_in) continue;

                    int input_idx = ((n * C_in + ic) * H_in + in_y) * W_in + in_x;
                    int weight_idx = ((ic * C_out + oc) * K + ky) * K + kx;  // weight shape: (C_in, C_out, K, K)
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    } else {
        // Handle grouped convolution
        int group_in_channels = C_in / groups;
        int group_out_channels = C_out / groups;
        int group = oc / group_out_channels;
        
        for (int ic_local = 0; ic_local < group_in_channels; ic_local++) {
            int ic = group * group_in_channels + ic_local;
            for (int ky = 0; ky < K; ky++) {
                int in_y = out_y + padding - ky;
                if (in_y % stride != 0) continue;
                in_y /= stride;
                if (in_y < 0 || in_y >= H_in) continue;
                
                for (int kx = 0; kx < K; kx++) {
                    int in_x = out_x + padding - kx;
                    if (in_x % stride != 0) continue;
                    in_x /= stride;
                    if (in_x < 0 || in_x >= W_in) continue;
                    
                    int input_idx = ((n * C_in + ic) * H_in + in_y) * W_in + in_x;
                    // For grouped conv, weight shape: (C_in, C_out/groups, K, K).
                    // Compute local output channel
                    int local_oc = oc - group * group_out_channels;
                    int weight_idx = ((ic_local * group_out_channels + local_oc) * K + ky) * K + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[index] = sum;
}


// Forward function callable from PyTorch
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    }

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Determine kernel size (assuming square kernel) and output channels
    int K = weight.size(2);  // For groups==1: weight shape (C_in, C_out, K, K)
    int C_out = (groups == 1) ? weight.size(1) : (weight.size(1) * groups);

    // Compute output spatial dimensions using the standard formula:
    // H_out = (H_in - 1) * stride - 2*padding + K + output_padding
    int H_out = (H_in - 1) * stride - 2 * padding + K + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K + output_padding;

    auto options = input.options();
    auto output_tensor = torch::zeros({N, C_out, H_out, W_out}, options);

    int total = N * C_out * H_out * W_out;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    conv_transpose2d_gather_forward<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K, stride, padding, groups
    );

    cudaDeviceSynchronize();
    return output_tensor;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Custom ConvTranspose2d forward (CUDA) - gather based, minimal atomics");
}
