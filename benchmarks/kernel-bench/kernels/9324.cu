#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void conv_transpose1d_kernel_shared(
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
    extern __shared__ float sweight[];
    int b = blockIdx.y;
    int oc = blockIdx.x;
    int weight_elements = in_channels * kernel_size;

    for (int idx = threadIdx.x; idx < weight_elements; idx += blockDim.x) {
        int k = idx / in_channels;
        int ic = idx % in_channels;
        sweight[k * in_channels + ic] = weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
    }
    __syncthreads();

    for (int o = threadIdx.x; o < output_length; o += blockDim.x) {
        float sum = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            int i_pos = o + padding - k * dilation;
            if (i_pos % stride != 0) continue;
            int i = i_pos / stride;
            if (i < 0 || i >= input_length) continue;

            #pragma unroll 4
            for (int ic = 0; ic < in_channels; ++ic) {
                float w = sweight[k * in_channels + ic];
                sum += x_ptr[b * in_channels * input_length + ic * input_length + i] * w;
            }
        }

        if (bias_ptr) sum += bias_ptr[oc];
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

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    dim3 grid(out_channels, batch_size);
    size_t shared_size = in_channels * kernel_size * sizeof(float);

    conv_transpose1d_kernel_shared<<<grid, 512, shared_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
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
    m.def("forward", &forward_cuda, "ConvTranspose1D_optimized");
}
