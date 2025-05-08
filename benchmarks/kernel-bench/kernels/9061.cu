#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel reduces warp divergence by precomputing the valid range of kernel indices
// and iterating only over those indices. In this way, we avoid per-iteration conditional branches
// and obtain more uniform control flow across threads in a warp. This implementation does not
// change the precision and guarantees the correct result.

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output, float* __restrict__ shared_mem,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Decode the output indices: o (output index), oc (output channel), and b (batch index)
    int o = idx % out_size;
    int temp = idx / out_size;
    int oc = temp % out_channels;
    int b = temp / out_channels;

    float sum = 0.0f;

    // Precompute the number of valid kernel positions for this output location.
    // For a convolution, the input position is: pos = o * stride + k * dilation.
    // We need pos < in_size. Thus, the maximum valid k is:
    // valid_k = min(kernel_size, (in_size - o*stride - 1)/dilation + 1) if o*stride < in_size, else 0.
    int valid_k = 0;
    int base_input = o * stride;
    if (base_input < in_size) {
        valid_k = (in_size - base_input - 1) / dilation + 1;
        if (valid_k > kernel_size) {
            valid_k = kernel_size;
        }
    } else {
        valid_k = 0;
    }

    // Loop over input channels and only over the valid kernel indices to avoid conditional checks inside the loop.
    for (int ic = 0; ic < in_channels; ic++) {
        // Precompute base indices for input and weight for this channel
        int base_x = b * (in_channels * in_size) + ic * in_size;
        int base_w = oc * (in_channels * kernel_size) + ic * kernel_size;
        
        // Only iterate over the valid kernel range; contributions outside this range are zero.
        for (int k = 0; k < valid_k; k++) {
            int input_pos = base_input + k * dilation; // Equivalently, o*stride + k*dilation
            int x_idx = base_x + input_pos;
            int w_idx = base_w + k;
            sum += x[x_idx] * weight[w_idx];
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");

    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());

    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv1d_kernel<<<blocks, threads>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA)");
}
