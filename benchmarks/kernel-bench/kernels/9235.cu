#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute the output length for the ConvTranspose1D operation
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel implementing ConvTranspose1D with branchless conditional logic
// to minimize warp divergence. Instead of early exits and divergent branches,
// we compute a validity mask for each kernel tap and use a clamped index for safe memory access.

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_length;
    if (idx >= total) return;

    // Decompose linear index into batch index, output channel, and spatial location
    int o  = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int b  = idx / (out_channels * output_length);

    float sum = 0.0f;
    // Loop over kernel positions without early exits. 
    // Instead, compute a validity mask for each tap in a branchless manner.
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        // Compute mask for valid tap based on stride alignment
        int mod_valid = ((i_pos % stride) == 0) ? 1 : 0;
        int i = i_pos / stride;
        // Compute mask for valid index range
        int range_valid = ((i >= 0) && (i < input_length)) ? 1 : 0;
        int valid = mod_valid * range_valid;  // 1 if valid; 0 otherwise

        // Clamp i to a safe index so that memory access is always valid
        int safe_i = i;
        safe_i = (i < 0) ? 0 : safe_i;
        safe_i = (i >= input_length) ? (input_length - 1) : safe_i;

        float inner_sum = 0.0f;
        // Iterate over input channels uniformly
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + safe_i;
            int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            // Use a ternary operator for branchless selection: if i is in range, read x; else use 0
            float x_val = ((i >= 0) && (i < input_length)) ? x_ptr[x_idx] : 0.0f;
            inner_sum += x_val * weight_ptr[weight_idx];
        }
        // Multiply the inner sum by the validity mask to include contributions only from valid taps
        sum += inner_sum * valid;
    }

    // Add bias if provided
    if (bias_ptr != nullptr) {
        sum += bias_ptr[oc];
    }

    int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

// Host function launching the CUDA kernel
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int total_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
