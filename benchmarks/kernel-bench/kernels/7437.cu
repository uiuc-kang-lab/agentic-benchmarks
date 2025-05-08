#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Maximum number of floats that can be stored in constant memory (64KB / sizeof(float) = 16384 floats)
#define MAX_WEIGHT_SIZE 16384

// Declare constant memory for the weight tensor.
// Expected layout: [C_in, C_out, kernel_size, kernel_size] in row-major order.
__constant__ float const_weight[MAX_WEIGHT_SIZE];

// Custom kernel to compute transposed convolution
// Assumes groups == 1
// x: [N, C_in, H_in, W_in]
// output: [N, C_out, H_out, W_out]
// weight is stored in constant memory
// bias: [C_out] (optional)
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode linear index into n, c_out, h_out, w_out
    int w_out = index % W_out;
    int temp = index / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = 0.0f;

    // Iterate over kernel elements to accumulate contributions
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int h_in_times_stride = h_out + padding - i;
            int w_in_times_stride = w_out + padding - j;
            // Check if these correspond to valid input positions (divisible by stride)
            if ((h_in_times_stride % stride == 0) && (w_in_times_stride % stride == 0)) {
                int h_in = h_in_times_stride / stride;
                int w_in = w_in_times_stride / stride;
                // Validate input boundaries
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // Sum contributions from all input channels
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        float x_val = x[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                        // Weight layout: [c_in, c_out, i, j]
                        int weight_index = ((c_in * C_out + c_out) * kernel_size + i) * kernel_size + j;
                        float w_val = const_weight[weight_index];
                        sum += x_val * w_val;
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    output[index] = sum;
}

// Forward function definition
// This function sets up the kernel launch and copies the weight tensor to constant memory
// Only supports groups == 1 and expects weight tensor size to fit within constant memory limits
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Check inputs
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    // This custom kernel supports only groups == 1
    TORCH_CHECK(groups == 1, "groups > 1 not supported in constant_memory_conv_transpose2d_optimized kernel");

    // Ensure weight fits in constant memory
    TORCH_CHECK(weight.numel() <= MAX_WEIGHT_SIZE, "Weight tensor exceeds constant memory capacity");

    // Retrieve input dimensions
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    
    // Assuming weight tensor shape is [C_in, C_out, kernel_size, kernel_size]
    int kernel_size = weight.size(2);  // square kernel assumed
    int C_out = weight.size(1);

    // Compute output dimensions for transposed convolution
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Copy weight tensor to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float));

    // Allocate output tensor
    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Prepare bias pointer if provided
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias.value().data_ptr<float>();
    }

    int total_output = N * C_out * H_out * W_out;
    const int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_size, stride, padding
    );
    cudaDeviceSynchronize();

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - constant memory optimized");
}
