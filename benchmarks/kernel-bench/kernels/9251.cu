#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void vectorized_conv_transpose1d_kernel(
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    const int o = idx % output_length;
    const int oc = (idx / output_length) % out_channels;
    const int b = idx / (out_channels * output_length);

    const int batch_offset = b * in_channels * input_length;
    const int output_batch_offset = b * out_channels * output_length;
    
    float sum = bias_ptr ? bias_ptr[oc] : 0.0f;

    const int kernel_aligned = kernel_size & ~3;
    
    #pragma unroll 4
    for (int k = 0; k < kernel_aligned; k += 4) {
        float4 partial_sums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        for (int ki = 0; ki < 4; ki++) {
            const int k_pos = k + ki;
            const int i_pos = o + padding - k_pos * dilation;
            
            if (i_pos % stride == 0) {
                const int i = i_pos / stride;
                if (i >= 0 && i < input_length) {
                    #pragma unroll
                    for (int ic = 0; ic < in_channels; ic++) {
                        const int x_idx = batch_offset + ic * input_length + i;
                        const int w_idx = ic * out_channels * kernel_size + oc * kernel_size + k_pos;
                        partial_sums.x += x_ptr[x_idx] * weight_ptr[w_idx];
                    }
                }
            }
        }
        
        sum += partial_sums.x + partial_sums.y + partial_sums.z + partial_sums.w;
    }

    for (int k = kernel_aligned; k < kernel_size; k++) {
        const int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        
        const int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        for (int ic = 0; ic < in_channels; ic++) {
            const int x_idx = batch_offset + ic * input_length + i;
            const int w_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            sum += x_ptr[x_idx] * weight_ptr[w_idx];
        }
    }

    output_ptr[output_batch_offset + oc * output_length + o] = sum;
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
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");

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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    const int num_elements = batch_size * out_channels * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    vectorized_conv_transpose1d_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}