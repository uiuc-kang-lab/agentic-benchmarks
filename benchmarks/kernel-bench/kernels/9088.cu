#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel removes conditional branches within the inner loop by leveraging that
// out_size is computed so that all accessed input positions are guaranteed valid for a valid convolution.

__global__ void conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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

    // Calculate output indices
    int o = idx % out_size;
    int temp = idx / out_size;
    int oc = temp % out_channels;
    int b = temp / out_channels;

    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    float sum = 0.0f;

    // Precompute offset for input and weight
    int b_offset = b * (in_channels * in_size);
    int oc_offset = oc * (in_channels * kernel_size);
    int base_input_idx = o * stride; // Guaranteed to be in range for valid convolution

    // No conditional check inside the loop to avoid warp divergence.
    for (int ic = 0; ic < in_channels; ++ic) {
        int x_offset = b_offset + ic * in_size;
        int w_offset = oc_offset + ic * kernel_size;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            // Compute input position, which is always valid given the computed out_size
            int input_pos = base_input_idx + k * dilation;
            sum += x[x_offset + input_pos] * weight[w_offset + k];
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

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

    // Compute output size for valid convolution (no padding)
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
    m.def("forward", &forward, "1D convolution forward (CUDA) - branchless version");
}
