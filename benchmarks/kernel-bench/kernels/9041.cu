#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum allowed sizes for constant memory (tuned for 64KB constant memory, approx 15360 floats & 1024 floats)
#define MAX_WEIGHT_SIZE 15360
#define MAX_BIAS_SIZE 1024

// Declare constant memory arrays for weight and bias
__constant__ float c_weight[MAX_WEIGHT_SIZE];
__constant__ float c_bias[MAX_BIAS_SIZE];

// Combined CUDA kernel that selects between constant memory and global memory for weight/bias
__global__ void conv1d_combined_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    bool use_const,
    bool use_bias_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Compute indices for output element
    int o = idx % out_size;
    idx /= out_size;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    float sum = 0.0f;
    
    // Iterate over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = o * stride + k * dilation;
            if (input_pos < in_size) {
                int x_idx = b * (in_channels * in_size) + ic * in_size + input_pos;
                int w_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                float weight_val = use_const ? c_weight[w_idx] : weight[w_idx];
                sum += x[x_idx] * weight_val;
            }
        }
    }
    
    // Add bias if needed
    if (use_bias_flag) {
        if (use_const) {
            sum += c_bias[oc];
        } else {
            sum += bias[oc];
        }
    }
    
    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
}


// Combined forward function: selects constant memory usage if weight (and bias) fit in available space
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    // Validate inputs
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, in_channels, in_size)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channel mismatch between x and weight");

    bool use_bias_flag = false;
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias_tensor.dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias_tensor.size(0) == weight.size(0), "Bias size must match number of output channels");
        use_bias_flag = true;
    }
    
    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size computed");

    // Decide whether to use constant memory based on tensor sizes
    bool use_const = (weight.numel() <= MAX_WEIGHT_SIZE) &&
                     ((!use_bias_flag) || (bias.value().numel() <= MAX_BIAS_SIZE));

    if (use_const) {
        size_t weight_bytes = weight.numel() * sizeof(float);
        cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_bytes, 0, cudaMemcpyDeviceToDevice);
        if (use_bias_flag) {
            size_t bias_bytes = bias.value().numel() * sizeof(float);
            cudaMemcpyToSymbol(c_bias, bias.value().data_ptr<float>(), bias_bytes, 0, cudaMemcpyDeviceToDevice);
        }
    }
    
    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = use_bias_flag ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv1d_combined_kernel<<<blocks, threads>>>(
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
        dilation,
        use_const,
        use_bias_flag
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined 1D convolution forward with constant memory optimization (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor(),
          pybind11::arg("stride"),
          pybind11::arg("dilation"));
}
