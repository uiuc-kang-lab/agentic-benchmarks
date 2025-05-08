#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum allowed sizes for constant memory storage
// 64KB constant memory is common; using 16384 floats (4 bytes each) equals 64KB
#define MAX_WEIGHT_SIZE 16384
#define MAX_BIAS_SIZE 1024

// Declare constant memory arrays for weight and bias
__constant__ float c_weight[MAX_WEIGHT_SIZE];
__constant__ float c_bias[MAX_BIAS_SIZE];

// CUDA kernel that uses constant memory for weight and bias
__global__ void conv1d_const_kernel(
    const float* __restrict__ x,
    float* output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    bool use_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

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
                sum += x[x_idx] * c_weight[w_idx];
            }
        }
    }
    
    if (use_bias) {
        sum += c_bias[oc];
    }
    
    int out_idx = b * (out_channels * out_size) + oc * out_size + o;
    output[out_idx] = sum;
}

// Forward function: loads weight (and bias) into constant memory and launches the kernel
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
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, in_channels, in_size)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channel mismatch between x and weight");

    bool use_bias = false;
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "Bias size must match number of output channels");
        use_bias = true;
    }
    
    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size computed");
    
    // Ensure weight (and bias) fit into constant memory
    TORCH_CHECK(weight.numel() <= MAX_WEIGHT_SIZE, "Weight tensor too large for constant memory");
    if (use_bias) {
        TORCH_CHECK(bias->numel() <= MAX_BIAS_SIZE, "Bias tensor too large for constant memory");
    }
    
    // Copy weight from device memory to constant memory
    size_t weight_bytes = weight.numel() * sizeof(float);
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_bytes, 0, cudaMemcpyDeviceToDevice);
    
    if (use_bias) {
        size_t bias_bytes = bias->numel() * sizeof(float);
        cudaMemcpyToSymbol(c_bias, bias->data_ptr<float>(), bias_bytes, 0, cudaMemcpyDeviceToDevice);
    }
    
    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;
    
    const float* x_data = x.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    int total_elements = B * out_channels * out_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv1d_const_kernel<<<blocks, threads>>>(
        x_data, 
        output_data, 
        B, 
        in_channels, 
        in_size, 
        out_channels, 
        kernel_size, 
        out_size, 
        stride, 
        dilation,
        use_bias
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with constant memory optimization (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = torch::Tensor(),
          pybind11::arg("stride"),
          pybind11::arg("dilation"));
}
