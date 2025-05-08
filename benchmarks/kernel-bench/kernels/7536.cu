#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_hybrid_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kT, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups,
    bool use_aten
) {
    if (use_aten) return;

    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_input = shared_mem;
    scalar_t* shared_weight = shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_channels * out_depth * out_height * out_width) return;

    int w = idx % out_width;
    int tmp = idx / out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int d = tmp % out_depth;
    tmp /= out_depth;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    int group = c / (out_channels / groups);
    int out_c_local = c % (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    scalar_t sum = 0;
    
    for (int ic_block = 0; ic_block < in_channels_per_group; ic_block += blockDim.x) {
        int ic = ic_block + threadIdx.x;
        if (ic < in_channels_per_group) {
            int input_channel = group * in_channels_per_group + ic;
            
            for (int k = 0; k < kT * kH * kW && threadIdx.x < blockDim.x; k++) {
                if (ic < in_channels_per_group) {
                    shared_weight[threadIdx.x] = weight[((input_channel * (out_channels / groups) + out_c_local) * kT * kH * kW) + k];
                }
            }
        }
        __syncthreads();

        if (idx < N * out_channels * out_depth * out_height * out_width) {
            for (int k = 0; k < min(blockDim.x, in_channels_per_group - ic_block); k++) {
                int ic = ic_block + k;
                int input_channel = group * in_channels_per_group + ic;

                for (int kd = 0; kd < kT; kd++) {
                    int d_in_tmp = d + pad_d - kd;
                    if (d_in_tmp % stride_d != 0) continue;
                    int d_in = d_in_tmp / stride_d;
                    if (d_in < 0 || d_in >= in_depth) continue;

                    for (int kh = 0; kh < kH; kh++) {
                        int h_in_tmp = h + pad_h - kh;
                        if (h_in_tmp % stride_h != 0) continue;
                        int h_in = h_in_tmp / stride_h;
                        if (h_in < 0 || h_in >= in_height) continue;

                        for (int kw = 0; kw < kW; kw++) {
                            int w_in_tmp = w + pad_w - kw;
                            if (w_in_tmp % stride_w != 0) continue;
                            int w_in = w_in_tmp / stride_w;
                            if (w_in < 0 || w_in >= in_width) continue;

                            int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                            int weight_idx = k * kT * kH * kW + (kd * kH + kh) * kW + kw;
                            
                            sum += input[input_idx] * shared_weight[weight_idx];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (idx < N * out_channels * out_depth * out_height * out_width) {
        if (bias != nullptr) {
            sum += bias[c];
        }
        output[idx] = sum;
    }
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
    if (input.numel() < 1024) {
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

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(scalar_t) * 2;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_hybrid_kernel", ([&] {
        transposed_conv3d_hybrid_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
            groups,
            false
        );
    }));

    return output;
}