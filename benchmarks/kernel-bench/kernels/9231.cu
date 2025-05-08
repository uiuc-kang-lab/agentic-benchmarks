#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Computes the length of the output for ConvTranspose1D.
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// This kernel assigns one thread per output element, fusing the improvements from the two kernels.
// It uses __restrict__ qualifiers, precomputed index arithmetic and optional bias handling to
// improve memory coalescing and reduce redundant computations.

__global__ void fast_conv_transpose1d_kernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_length;
    if (idx >= total) return;

    // Decompose linear index to get batch, output channel, and spatial index
    int o = idx % output_length;
    int temp = idx / output_length;
    int oc = temp % out_channels;
    int b = temp / out_channels;

    // Compute offsets to access the output memory
    int out_offset = b * (out_channels * output_length) + oc * output_length;
    
    // Each sample in x has 'in_channels * input_length' elements
    int x_batch_stride = in_channels * input_length;

    // Initialize the accumulated sum; add bias if available
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Precompute padded output index
    int o_pad = o + padding;

    // Loop over the kernel positions
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o_pad - k * dilation;
        // Check if the position aligns with the stride
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        // Loop over the input channels
        int x_base = b * x_batch_stride;
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = x_base + ic * input_length + i;
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x[x_idx] * weight[weight_idx];
        }
    }

    output[out_offset + o] = sum;
}

// Host function which sets up the tensors and launches the CUDA kernel
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
    TORCH_CHECK(weight.size(0) == in_channels, "Weight in_channels mismatch.");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "Bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int total_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fast_conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward_cuda, "Fast ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
