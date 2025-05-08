#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum sizes for constant memory (in floats)
#define MAX_WEIGHT_SIZE 15360  // 15360 * 4 bytes = 60KB
#define MAX_BIAS_SIZE 1024    // 1024 * 4 bytes = 4KB

// Declare weight and bias in constant memory
__constant__ float c_weight[MAX_WEIGHT_SIZE];
__constant__ float c_bias[MAX_BIAS_SIZE];

// Helper to compute output length
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel using constant memory for weight and bias
__global__ void conv_transpose1d_kernel(
    const float* x_ptr,
    float* output_ptr,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float sum = 0.0f;

    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        // Loop over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x_ptr[x_idx] * c_weight[weight_idx];
        }
    }
    if (has_bias) {
        sum += c_bias[oc];
    }
    int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

// Forward function that copies weight (and bias) to constant memory and launches the kernel
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

    int weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_WEIGHT_SIZE, "Weight tensor exceeds constant memory capacity");
    
    bool bias_flag = false;
    torch::Tensor bias_contig;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        int out_channels = weight.size(1);
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
        TORCH_CHECK(out_channels <= MAX_BIAS_SIZE, "Bias tensor exceeds constant memory capacity");
        bias_flag = true;
    }

    // Copy weight (and bias if present) to constant memory
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));
    if (bias_flag) {
        int out_channels = weight.size(1);
        cudaMemcpyToSymbol(c_bias, bias_contig.data_ptr<float>(), out_channels * sizeof(float));
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int num_output_elements = batch_size * out_channels * output_length;
    int threads_per_block = 512;
    int num_blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        bias_flag ? 1 : 0
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with constant memory",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
