#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void optimized_conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    extern __shared__ float shared_weight[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= batch_size * out_channels * output_length) return;

    const int b = idx / (out_channels * output_length);
    const int rem = idx % (out_channels * output_length);
    const int oc = rem / output_length;
    const int o = rem % output_length;

    const int weights_per_thread = (in_channels * kernel_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; i++) {
        const int w_idx = tid * weights_per_thread + i;
        if (w_idx < in_channels * kernel_size) {
            shared_weight[w_idx] = __ldg(weight + (w_idx / kernel_size) * out_channels * kernel_size 
                                   + oc * kernel_size + (w_idx % kernel_size));
        }
    }
    __syncthreads();

    float sum = 0.0f;
    const int o_padded = o + padding;
    const int input_stride = input_length;
    const int batch_stride = in_channels * input_length;

    #pragma unroll 4
    for (int k = 0; k < kernel_size; k++) {
        const int i_pos = o_padded - k * dilation;
        if (i_pos % stride == 0) {
            const int i = i_pos / stride;
            if (i >= 0 && i < input_length) {
                const float* x_ptr = x + b * batch_stride + i;
                #pragma unroll 2
                for (int ic = 0; ic < in_channels; ic++) {
                    sum += __ldg(x_ptr + ic * input_stride) * 
                           shared_weight[ic * kernel_size + k];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __ldg(bias + oc);
    }

    output[idx] = sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    x = x.contiguous();
    weight = weight.contiguous();
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * output_length;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const int shared_mem_size = in_channels * kernel_size * sizeof(float);

    optimized_conv_transpose1d_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
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