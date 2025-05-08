#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_size = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int idx = tid; idx < total_elements; idx += stride_size) {
        const int b = idx / (out_channels * output_length);
        const int rem = idx % (out_channels * output_length);
        const int oc = rem / output_length;
        const int o = rem % output_length;
        
        float sum = 0.0f;
        const int b_offset = b * in_channels * input_length;
        const int oc_offset = oc * kernel_size;
        
        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            const int i_pos = o + padding - k * dilation;
            if (i_pos % stride == 0) {
                const int i = i_pos / stride;
                if (i >= 0 && i < input_length) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int x_idx = b_offset + ic * input_length + i;
                        const int w_idx = ic * out_channels * kernel_size + oc_offset + k;
                        sum += x_ptr[x_idx] * weight_ptr[w_idx];
                    }
                }
            }
        }
        
        if (bias_ptr != nullptr) {
            sum += bias_ptr[oc];
        }
        
        output_ptr[b * out_channels * output_length + oc * output_length + o] = sum;
    }
}

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
    
    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());
    
    const int total_elements = batch_size * out_channels * output_length;
    
    // Optimize block and grid size for better occupancy
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    const int num_blocks = min((total_elements + threads_per_block - 1) / threads_per_block, max_blocks);
    
    conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
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
        dilation,
        total_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}