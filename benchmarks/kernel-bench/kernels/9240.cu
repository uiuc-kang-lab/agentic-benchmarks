#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute the output length for ConvTranspose1D
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// This kernel leverages shared memory to cache the weight tensor, reducing global memory latency
// for frequently reused weight data in the transposed 1D convolution. Each block cooperatively loads
// the weight values into shared memory before processing its assigned output elements.

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
    // Allocate shared memory for the weight tensor
    extern __shared__ float s_weight[];
    int weight_count = in_channels * out_channels * kernel_size;
    
    // Loading the full weight tensor into shared memory, done cooperatively by all threads in the block
    for (int i = threadIdx.x; i < weight_count; i += blockDim.x) {
        s_weight[i] = weight_ptr[i];
    }
    __syncthreads();

    // Compute the global index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    // Decompose the linear index into batch, output channel, and spatial index
    int o = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int b = idx / (out_channels * output_length);

    float sum = 0.0f;
    int o_pad = o + padding;  // account for padding

    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o_pad - k * dilation;
        if (i_pos % stride != 0) continue;  // only consider valid positions
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        // Each input channel contributes to the output
        int x_base = b * in_channels * input_length;
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = x_base + ic * input_length + i;
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x_ptr[x_idx] * s_weight[weight_idx];
        }
    }

    // Add bias if available
    if (bias_ptr) {
        sum += bias_ptr[oc];
    }

    int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

// Host function: sets up tensors, calculates dimensions and launches the CUDA kernel
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

    int num_output_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    int num_blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;

    // Calculate the size needed for shared memory to hold the entire weight tensor
    int shared_mem_size = in_channels * out_channels * kernel_size * sizeof(float);

    conv_transpose1d_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with shared memory optimization",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
