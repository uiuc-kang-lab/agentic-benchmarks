#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

template<int KERNEL_SIZE=3>
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
    const int stride,
    const int padding,
    const int dilation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    const int o = idx % output_length;
    const int oc = (idx / output_length) % out_channels;
    const int b = idx / (out_channels * output_length);

    float sum = 0.0f;
    
    // Manual unroll for kernel size
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        const int i_pos = o + padding - k * dilation;
        if (i_pos % stride == 0) {
            const int i = i_pos / stride;
            if (i >= 0 && i < input_length) {
                // Partial unroll for input channels
                const int ic_unroll = 4;
                const int ic_limit = in_channels / ic_unroll * ic_unroll;
                const int base_x_offset = b * in_channels * input_length + i;
                const int base_w_offset = oc * kernel_size + k;

                // Process 4 input channels at a time
                #pragma unroll
                for (int ic = 0; ic < ic_limit; ic += ic_unroll) {
                    float sum0 = x_ptr[base_x_offset + (ic + 0) * input_length] * 
                               weight_ptr[(ic + 0) * out_channels * KERNEL_SIZE + base_w_offset];
                    float sum1 = x_ptr[base_x_offset + (ic + 1) * input_length] * 
                               weight_ptr[(ic + 1) * out_channels * KERNEL_SIZE + base_w_offset];
                    float sum2 = x_ptr[base_x_offset + (ic + 2) * input_length] * 
                               weight_ptr[(ic + 2) * out_channels * KERNEL_SIZE + base_w_offset];
                    float sum3 = x_ptr[base_x_offset + (ic + 3) * input_length] * 
                               weight_ptr[(ic + 3) * out_channels * KERNEL_SIZE + base_w_offset];
                    sum += sum0 + sum1 + sum2 + sum3;
                }

                // Handle remaining channels
                for (int ic = ic_limit; ic < in_channels; ++ic) {
                    const int x_idx = base_x_offset + ic * input_length;
                    const int weight_idx = ic * out_channels * KERNEL_SIZE + base_w_offset;
                    sum += x_ptr[x_idx] * weight_ptr[weight_idx];
                }
            }
        }
    }

    if (bias_ptr != nullptr) {
        sum += bias_ptr[oc];
    }

    output_ptr[b * out_channels * output_length + oc * output_length + o] = sum;
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
        bias_ptr = bias_contig.data_ptr<float>();
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    const int num_elements = batch_size * out_channels * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    if (kernel_size == 3) {
        conv_transpose1d_kernel<3><<<num_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            stride,
            padding,
            dilation
        );
    } else {
        conv_transpose1d_kernel<0><<<num_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            stride,
            padding,
            dilation
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}