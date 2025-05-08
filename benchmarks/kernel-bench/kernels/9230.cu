/*
 * Optimized ConvTranspose1d CUDA kernel
 * Combines improved memory coalescing with __restrict__ qualifiers and __ldg() for read-only loads.
 * This kernel assigns one thread per output element and decodes the linear index to (batch, out_channel, output_index) for coalesced global memory accesses.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute output length for transposed convolution
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Optimized kernel: combines memory coalescing with __ldg for read-only access
__global__ void optimized_conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // Can be nullptr if not provided
    float* __restrict__ output,
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

    // Decode linear index to (batch, output channel, output spatial index)
    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float sum = 0.0f;
    int o_padded = o + padding;

    // Loop over kernel positions and input channels
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o_padded - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            sum += __ldg(&x[x_idx]) * __ldg(&weight[weight_idx]);
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    // Coalesced write using linear index
    output[idx] = sum;
}

// Host launcher function

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Compute output spatial length
    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int total_output_elements = batch_size * out_channels * output_length;
    const int threads_per_block = 256;
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;

    optimized_conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward_cuda, "Optimized ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
