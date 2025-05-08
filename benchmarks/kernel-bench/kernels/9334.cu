#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute the output length given convolution parameters
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// Modular device function that computes the convolution transpose contribution for a single output position
__device__ inline float compute_output_for_position(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    int b,
    int oc,
    int o,
    int in_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int out_channels
) {
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = b * in_channels * input_length + ic * input_length + i;
            int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            sum += x_ptr[x_idx] * weight_ptr[weight_idx];
        }
    }
    return sum;
}

// Kernel that utilizes the modular device function for better readability and maintainability
__global__ void conv_transpose1d_kernel_shared(
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
    // Each block processes a single (batch, out_channel) pair
    int b = blockIdx.x / out_channels;
    int oc = blockIdx.x % out_channels;
    int tid = threadIdx.x;

    extern __shared__ float shared_weight[]; // size: in_channels * kernel_size floats

    // Load weight for the current out_channel into shared memory
    for (int idx = tid; idx < in_channels * kernel_size; idx += blockDim.x) {
        int ic = idx / kernel_size;
        int k = idx % kernel_size;
        shared_weight[idx] = weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
    }
    __syncthreads();

    // Each thread processes multiple output positions
    for (int o = tid; o < output_length; o += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int i_pos = o + padding - k * dilation;
            if (i_pos % stride != 0) continue;
            int i = i_pos / stride;
            if (i < 0 || i >= input_length) continue;
            for (int ic = 0; ic < in_channels; ic++) {
                int x_idx = b * in_channels * input_length + ic * input_length + i;
                int weight_idx = ic * kernel_size + k; // access from shared memory
                sum += x_ptr[x_idx] * shared_weight[weight_idx];
            }
        }
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        int out_idx = b * out_channels * output_length + oc * output_length + o;
        output_ptr[out_idx] = sum;
    }
}

// Forward function called from PyTorch
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

    conv_transpose1d_kernel_modular<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with modular device functions",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
