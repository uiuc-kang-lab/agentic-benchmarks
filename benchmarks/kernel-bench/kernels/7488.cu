#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Optimized CUDA kernel for transposed convolution with square input and square kernel
// This kernel uses __ldg for read-only global loads and assumes that memory is 128-bit aligned.

// __global__ function definition
__global__ void convTranspose2dKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int H_out,
    int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * H_out * W_out;
    if (index >= total) return;

    // Decode the linear index into (n, oc, oh, ow)
    int ow = index % W_out;
    int temp = index / W_out;
    int oh = temp % H_out;
    temp = temp / H_out;
    int oc = temp % out_channels;
    int n = temp / out_channels;

    float value = 0.0f;
    // Loop over input channels and kernel spatial positions
    // For transposed convolution, output(n, oc, oh, ow) is computed by accumulating contributions
    // from input(n, ic, i_in, j_in) and weight(ic, oc, p, q) where:
    //   i_in = (oh + padding - p) / stride, provided (oh + padding - p) is divisible by stride
    //   j_in = (ow + padding - q) / stride, provided (ow + padding - q) is divisible by stride
    for (int ic = 0; ic < in_channels; ic++) {
        for (int p = 0; p < kernel_size; p++) {
            int h_offset = oh + padding - p;
            if (h_offset < 0) continue;
            if (h_offset % stride != 0) continue;
            int i_in = h_offset / stride;
            if (i_in >= H_in) continue;
            for (int q = 0; q < kernel_size; q++) {
                int w_offset = ow + padding - q;
                if (w_offset < 0) continue;
                if (w_offset % stride != 0) continue;
                int j_in = w_offset / stride;
                if (j_in >= W_in) continue;
                // Calculate indices assuming contiguous layout
                int input_index = ((n * in_channels + ic) * H_in + i_in) * W_in + j_in;
                int weight_index = ((ic * out_channels + oc) * kernel_size + p) * kernel_size + q;
                value += __ldg(&input[input_index]) * __ldg(&weight[weight_index]);
            }
        }
    }
    
    // Add bias if provided
    if (bias) {
        value += __ldg(&bias[oc]);
    }
    
    int output_index = ((n * out_channels + oc) * H_out + oh) * W_out + ow;
    output[output_index] = value;
}

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // This implementation currently supports groups == 1 only
    TORCH_CHECK(groups == 1, "conv_transposed2d_optimized only supports groups==1.");
    
    // Ensure inputs are on CUDA and are contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
    }

    // x shape: [batch_size, in_channels, H_in, W_in]
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    
    // weight shape: [in_channels, out_channels, kernel_size, kernel_size] (square kernel assumed)
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1);
    
    // Compute output dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, H_out, W_out}, options);

    int total_threads = batch_size * out_channels * H_out * W_out;
    int threads = 512;
    int blocks = (total_threads + threads - 1) / threads;

    // Get raw pointers from tensors
    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias_tensor.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Launch the CUDA kernel
    convTranspose2dKernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        H_in,
        W_in,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) optimized");
}
