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
    const int dilation
) {
    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_channels * output_length) return;

    // Decompose index for better memory access patterns
    const int o = tid % output_length;
    const int oc = (tid / output_length) % out_channels;
    const int b = tid / (out_channels * output_length);

    // Pre-calculate batch offset
    const int batch_offset = b * in_channels * input_length;
    const int weight_oc_offset = oc * kernel_size;
    
    // Initialize sum with bias if present
    float sum = (bias_ptr != nullptr) ? bias_ptr[oc] : 0.0f;

    // Pre-calculate the padded position
    const int o_pad = o + padding;

    // Main computation loop
    #pragma unroll 4
    for (int k = 0; k < kernel_size; ++k) {
        const int i_pos = o_pad - k * dilation;
        if (i_pos % stride == 0) {
            const int i = i_pos / stride;
            if (i >= 0 && i < input_length) {
                // Pre-calculate weight index base for current kernel position
                const int weight_k_offset = k;
                
                #pragma unroll 4
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int x_idx = batch_offset + ic * input_length + i;
                    const int weight_idx = ic * out_channels * kernel_size + weight_oc_offset + weight_k_offset;
                    sum += x_ptr[x_idx] * weight_ptr[weight_idx];
                }
            }
        }
    }

    // Write result directly to output (no atomic needed as each thread writes to unique location)
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
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;

    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cuda(), "bias must be a CUDA tensor");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Optimize thread block size for H100
    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_length;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

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
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}