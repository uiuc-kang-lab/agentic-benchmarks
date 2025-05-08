#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int OPTIMIZED_KERNEL_THRESHOLD = 64;

template <typename scalar_t>
__global__ void transposed_conv3d_optimized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int in_channels, const int in_depth, const int in_height, const int in_width,
    const int out_channels, const int out_depth, const int out_height, const int out_width,
    const int kT, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int out_pad_d, const int out_pad_h, const int out_pad_w,
    const int groups
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_weight = (scalar_t*)shared_mem;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * out_channels * out_depth * out_height * out_width;
    if (idx >= total) return;

    const int w = idx % out_width;
    int tmp = idx / out_width;
    const int h = tmp % out_height;
    tmp /= out_height;
    const int d = tmp % out_depth;
    tmp /= out_depth;
    const int c = tmp % out_channels;
    const int n = tmp / out_channels;

    const int group = c / (out_channels / groups);
    const int out_c_local = c % (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    for (int i = tid; i < in_channels_per_group * kT * kH * kW; i += block_size) {
        shared_weight[i] = weight[i + group * in_channels_per_group * kT * kH * kW];
    }
    __syncthreads();

    scalar_t sum = 0;

    #pragma unroll 4
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int input_channel = group * in_channels_per_group + ic;
        
        #pragma unroll 2
        for (int kd = 0; kd < kT; kd++) {
            const int d_in_tmp = d + pad_d - kd;
            if (d_in_tmp % stride_d != 0) continue;
            const int d_in = d_in_tmp / stride_d;
            if (d_in < 0 || d_in >= in_depth) continue;

            #pragma unroll 2
            for (int kh = 0; kh < kH; kh++) {
                const int h_in_tmp = h + pad_h - kh;
                if (h_in_tmp % stride_h != 0) continue;
                const int h_in = h_in_tmp / stride_h;
                if (h_in < 0 || h_in >= in_height) continue;

                #pragma unroll 2
                for (int kw = 0; kw < kW; kw++) {
                    const int w_in_tmp = w + pad_w - kw;
                    if (w_in_tmp % stride_w != 0) continue;
                    const int w_in = w_in_tmp / stride_w;
                    if (w_in < 0 || w_in >= in_width) continue;

                    const int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                    const int weight_idx = ((ic * (out_channels / groups) + out_c_local) * kT + kd) * kH * kW + kh * kW + kw;
                    
                    sum += input[input_idx] * shared_weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    output[idx] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    if (input.size(1) <= OPTIMIZED_KERNEL_THRESHOLD) {
        return at::conv_transpose3d(
            input,
            weight,
            bias ? *bias : torch::Tensor(),
            stride,
            padding,
            output_padding,
            groups,
            std::vector<int64_t>{1, 1, 1}
        );
    }

    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    const int out_channels = weight.size(1) * groups;
    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    const int threads = 256;
    const int total_elements = N * out_channels * out_depth * out_height * out_width;
    const int blocks = (total_elements + threads - 1) / threads;
    const int shared_mem_size = (in_channels / groups) * kT * kH * kW * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_optimized_kernel", ([&] {
        transposed_conv3d_optimized_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups
        );
    }));

    return output;
}