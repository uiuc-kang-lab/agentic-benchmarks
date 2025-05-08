#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper function to compute output length based on convolution parameters
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CUDA kernel using shared memory to load the weight tensor for faster access
__global__ void conv_transpose1d_kernel(
    const float* x_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    float* output_ptr,
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
    // Calculate the total number of weight elements
    int weight_size = in_channels * out_channels * kernel_size;
    
    // Allocate shared memory for weight copy
    extern __shared__ float shared_weight[];

    // Each thread loads a portion of the weight tensor into shared memory
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        shared_weight[i] = weight_ptr[i];
    }
    __syncthreads();

    // Compute global thread index for output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    // Map the thread index to (batch, out_channel, output position)
    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    float sum = 0.0f;
    
    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        // Only consider valid positions that align with the stride
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;
        
        // Accumulate over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            // Weight index: [ic, oc, k] stored in contiguous order
            int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += x_ptr[x_idx] * shared_weight[weight_idx];
        }
    }

    // Add bias if provided
    if (bias_ptr) {
        sum += bias_ptr[oc];
    }

    // Write result to output tensor
    int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

// CUDA forward function wrapped for pybind11 binding
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

    // Calculate required shared memory size for weight tensor
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) using shared memory for weights",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
