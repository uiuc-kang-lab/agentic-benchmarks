#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute the output length for the transposed convolution
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Kernel using shared memory to cache weights with minimal synchronizations
// __syncthreads() is used only once after loading weights
__global__ void conv_transpose1d_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ global_weight,
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
    // Declare shared memory for weight caching
    extern __shared__ float shared_weight[];

    // Total number of weight elements
    int weight_size = in_channels * out_channels * kernel_size;
    
    // Each thread in the block loads part of the weight matrix
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        shared_weight[i] = global_weight[i];
    }
    // Synchronize once to ensure the shared memory is fully populated
    __syncthreads();

    // Compute global thread index for output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_length;
    if (idx >= total_elements) return;

    // Decode flat index into (batch, output_channel, output_position)
    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float acc = 0.0f;
    int o_padded = o + padding;

    // Loop over the kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int pos = o_padded - k * dilation;
        if (pos % stride != 0) continue;
        int i = pos / stride;
        if (i < 0 || i >= input_length) continue;

        // For each input channel
        int input_base = b * in_channels * input_length;
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = input_base + ic * input_length + i;
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            acc += x[x_idx] * shared_weight[weight_idx];
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        acc += bias[oc];
    }

    // Write result to the output tensor
    int out_idx = b * out_channels * output_length + oc * output_length + o;
    output[out_idx] = acc;
}

// Host function to launch the CUDA kernel
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

    TORCH_CHECK(weight.size(0) == in_channels, "Weight in_channels must match input channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "Bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int total_output_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = in_channels * out_channels * kernel_size * sizeof(float);

    conv_transpose1d_kernel_shared<<<num_blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward with shared memory and minimal synchronization (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
