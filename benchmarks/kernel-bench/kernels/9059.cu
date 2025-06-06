#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
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

    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Precompute base indices for x and weight for this channel
        int x_base = b * (in_channels * in_size) + ic * in_size;
        int w_base = oc * (in_channels * kernel_size) + ic * kernel_size;

        // Unrolling the inner loop over the kernel elements
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos >= 0 && input_pos < in_size) {
                sum += x[x_base + input_pos] * weight[w_base + k];
            }
        }
    }

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
    // Split the work into smaller chunks for better concurrent execution
    const int max_blocks_per_chunk = 65535; // Maximum blocks per grid dimension
    const int threads = 256;
    
    // Calculate total blocks needed
    int total_blocks = (total_elements + threads - 1) / threads;
    
    // Split into multiple kernel launches if needed
    int remaining_blocks = total_blocks;
    int offset = 0;
    
    while (remaining_blocks > 0) {
        int current_blocks = min(remaining_blocks, max_blocks_per_chunk);
        
        conv1d_kernel<<<current_blocks, threads>>>(
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
        
        // Update for next iteration
        remaining_blocks -= current_blocks;
        offset += current_blocks * threads;
        
        // Check for errors after each launch
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
    }

    // Remove the final error check since we're checking after each launch
    return output;

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
    m.def("forward", &forward, "1D convolution forward (CUDA) with unrolled inner loop");
}
