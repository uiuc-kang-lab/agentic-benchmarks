#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using thread stride loops to cover all output elements
__global__ void convTranspose2dStrideKernel(
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
    int W_out) {

    // Total number of output elements
    int total = batch * out_channels * H_out * W_out;
    
    // Each thread processes multiple outputs using a stride loop
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_val = blockDim.x * gridDim.x;
    
    for (int index = tid; index < total; index += stride_val) {
        // Decode the linear index into (n, oc, oh, ow)
        int ow = index % W_out;
        int temp = index / W_out;
        int oh = temp % H_out;
        temp /= H_out;
        int oc = temp % out_channels;
        int n = temp / out_channels;
        
        float acc = 0.0f;
        
        // Loop over input channels and kernel spatial positions
        for (int ic = 0; ic < in_channels; ic++) {
            for (int p = 0; p < kernel_size; p++) {
                int h_offset = oh + padding - p;
                if (h_offset < 0 || (h_offset % stride) != 0) continue;
                int i_in = h_offset / stride;
                if (i_in < 0 || i_in >= H_in) continue;
                
                for (int q = 0; q < kernel_size; q++) {
                    int w_offset = ow + padding - q;
                    if (w_offset < 0 || (w_offset % stride) != 0) continue;
                    int j_in = w_offset / stride;
                    if (j_in < 0 || j_in >= W_in) continue;
                    
                    int input_idx = ((n * in_channels + ic) * H_in + i_in) * W_in + j_in;
                    int weight_idx = ((ic * out_channels + oc) * kernel_size + p) * kernel_size + q;
                    acc += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
        
        // Add bias if provided
        if (bias) {
            acc += __ldg(&bias[oc]);
        }
        
        int output_idx = ((n * out_channels + oc) * H_out + oh) * W_out + ow;
        output[output_idx] = acc;
    }
}

// Forward function for ConvTranspose2d using the stride loop kernel
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Only support groups == 1
    TORCH_CHECK(groups == 1, "conv_transposed2d_stride_loops supports groups == 1 only.");
    
    // Ensure input tensors are on CUDA and contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    // Dimensions from input tensor
    int batch = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Weight shape: [in_channels, out_channels, kernel_size, kernel_size] (square kernel assumed)
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1);

    // Compute output dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor output = torch::zeros({batch, out_channels, H_out, W_out}, options);
    
    // Calculate total number of output elements for the kernel loop
    int total_elements = batch * out_channels * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // Launch the kernel with stride loops
    convTranspose2dStrideKernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
        W_out
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward with stride loops (CUDA)");
}
