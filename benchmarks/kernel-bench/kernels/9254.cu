#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Computes the output length for ConvTranspose1D
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel using 3D grid indexing
// Grid dimensions:
//   grid.z: batch index
//   grid.y: output channel index
//   grid.x: covers the output spatial dimension
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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
    // Determine output coordinate using 3D grid
    int b  = blockIdx.z;                  // Batch index
    int oc = blockIdx.y;                  // Output channel index
    int o  = blockIdx.x * blockDim.x + threadIdx.x;  // Spatial index in output
    if (o >= output_length) return;

    float sum = 0.0f;
    
    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        // Sum over all input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x[x_idx] * weight[weight_idx];
        }
    }

    // Add bias if provided
    if (bias) {
        sum += bias[oc];
    }

    int output_idx = b * (out_channels * output_length) + oc * output_length + o;
    output[output_idx] = sum;
}

// Host function launching the CUDA kernel with 3D grid indexing
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
    
    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Define 3D grid: x-dim covers the output spatial dimension,
    // y-dim for output channels, and z-dim for batch size
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    dim3 grid((output_length + block.x - 1) / block.x, out_channels, batch_size);

    conv_transpose1d_kernel<<<grid, block>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with 3D grid indexing",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
