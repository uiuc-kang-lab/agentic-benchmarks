#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_stride;
__constant__ int d_padding;
__constant__ int d_dilation;

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

template<int KERNEL_SIZE, int IN_CHANNELS>
__global__ void conv_transpose1d_kernel_small(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    const int batch_size,
    const int out_channels,
    const int input_length,
    const int output_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    const int o = idx % output_length;
    const int oc = (idx / output_length) % out_channels;
    const int b = idx / (out_channels * output_length);

    float sum = 0.0f;

    // Manual unroll for small kernel sizes
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        const int i_pos = o + d_padding - k * d_dilation;
        if (i_pos % d_stride == 0) {
            const int i = i_pos / d_stride;
            if (i >= 0 && i < input_length) {
                // Manual unroll for small channel counts
                #pragma unroll
                for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                    const int x_idx = b * IN_CHANNELS * input_length + ic * input_length + i;
                    const int weight_idx = ic * out_channels * KERNEL_SIZE + oc * KERNEL_SIZE + k;
                    sum += x_ptr[x_idx] * weight_ptr[weight_idx];
                }
            }
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[oc];
    }

    const int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
}

__global__ void conv_transpose1d_kernel_generic(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int output_length,
    const int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    const int o = idx % output_length;
    const int oc = (idx / output_length) % out_channels;
    const int b = idx / (out_channels * output_length);

    float sum = 0.0f;

    #pragma unroll 8
    for (int k = 0; k < kernel_size; ++k) {
        const int i_pos = o + d_padding - k * d_dilation;
        if (i_pos % d_stride == 0) {
            const int i = i_pos / d_stride;
            if (i >= 0 && i < input_length) {
                #pragma unroll 4
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int x_idx = b * in_channels * input_length + ic * input_length + i;
                    const int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
                    sum += x_ptr[x_idx] * weight_ptr[weight_idx];
                }
            }
        }
    }

    if (bias_ptr) {
        sum += bias_ptr[oc];
    }

    const int output_idx = b * out_channels * output_length + oc * output_length + o;
    output_ptr[output_idx] = sum;
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
    
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
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

    // Copy constants to device constant memory
    cudaMemcpyToSymbol(d_stride, &stride, sizeof(int));
    cudaMemcpyToSymbol(d_padding, &padding, sizeof(int));
    cudaMemcpyToSymbol(d_dilation, &dilation, sizeof(int));

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_length;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Use specialized kernel for common small sizes
    if (kernel_size == 3 && in_channels == 64) {
        conv_transpose1d_kernel_small<3, 64><<<num_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            out_channels,
            input_length,
            output_length
        );
    } else {
        conv_transpose1d_kernel_generic<<<num_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}