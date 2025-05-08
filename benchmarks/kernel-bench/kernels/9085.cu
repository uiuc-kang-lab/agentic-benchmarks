#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Ensure coalesced memory access pattern
    const int o = idx % out_size;
    const int oc = (idx / out_size) % out_channels;
    const int b = idx / (out_channels * out_size);

    float sum = 0.0f;
    
    // Pre-compute base indices for better memory access patterns
    const int batch_offset = b * (in_channels * in_size);
    const int weight_channel_offset = oc * (in_channels * kernel_size);
    
    // Process 4 elements at a time when possible for better memory alignment
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ++ic) {
        const int input_offset = batch_offset + ic * in_size;
        const int weight_offset = weight_channel_offset + ic * kernel_size;
        
        const int base_input_pos = o * stride;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int input_pos = base_input_pos + k * dilation;
            if (input_pos < in_size) {
                // Use __ldg for read-only data
                const float x_val = __ldg(&x[input_offset + input_pos]);
                const float w_val = __ldg(&weight[weight_offset + k]);
                sum += x_val * w_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(&bias[oc]);
    }

    // Ensure aligned store
    const int out_idx = b * (out_channels * out_size) + oc * out_size + o;
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

    const int B = x.size(0);
    const int in_channels = x.size(1);
    const int in_size = x.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias ? bias->data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Optimize thread block size for better occupancy
    constexpr int threads = 256;
    const int blocks = (B * out_channels * out_size + threads - 1) / threads;

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