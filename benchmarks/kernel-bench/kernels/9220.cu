#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute output length on the host side as well
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Improved kernel that assigns one thread per output element (gather approach).
// This avoids race conditions and thus does not require atomic operations.
// We add __restrict__ qualifiers to help the compiler optimize memory accesses.

__global__ void improved_conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
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
    // Each thread computes one output element identified by (b, oc, o)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_length;
    if (idx >= total) return;

    // Decompose linear index into (batch, output channel, output spatial index)
    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float sum = 0.0f;
    // Precompute o+p to save redundant additions
    int o_padded = o + padding;

    // Iterate over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        // Compute the corresponding position in the input that could contribute
        int i_pos = o_padded - k * dilation;
        // Check if i_pos is aligned with the stride
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        // For this kernel position, sum over all input channels
        // Compute base offset for input and weight
        int input_base = b * (in_channels * input_length);
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = input_base + ic * input_length + i;
            // Weight is stored as [in_channels, out_channels, kernel_size]
            int w_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x[x_idx] * weight[w_idx];
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write the result to the output tensor
    int out_base = b * (out_channels * output_length);
    output[out_base + oc * output_length + o] = sum;
}

// Host function to launch the improved kernel

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

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int total_output_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;

    improved_conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward_cuda, "Improved ConvTranspose1D forward (CUDA)\n",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}

// Example runtime (for reference):
// runtime: 0.01800 milliseconds
// speedup over torch: 0.85x
