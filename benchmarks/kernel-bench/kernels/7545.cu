#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_warp_shuffle_kernel(
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
    int groups
) {
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    
    if (warp_id >= N * out_channels * out_depth * out_height * out_width / warp_size) return;

    // Decompose warp_id to find output position
    int tmp = warp_id * warp_size + lane_id;
    const int w = tmp % out_width;
    tmp /= out_width;
    const int h = tmp % out_height;
    tmp /= out_height;
    const int d = tmp % out_depth;
    tmp /= out_depth;
    const int c = tmp % out_channels;
    const int n = tmp / out_channels;

    if (n >= N || c >= out_channels || d >= out_depth || h >= out_height || w >= out_width) return;

    const int group = c / (out_channels / groups);
    const int out_c_local = c % (out_channels / groups);
    const int in_channels_per_group = in_channels / groups;

    scalar_t sum = 0;

    // Each thread in warp processes subset of input channels
    for (int ic = lane_id; ic < in_channels_per_group; ic += warp_size) {
        const int input_channel = group * in_channels_per_group + ic;
        
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

                    const int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                    const int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in warp writes result
    if (lane_id == 0 && n < N && c < out_channels && d < out_depth && h < out_height && w < out_width) {
        const int out_idx = (((n * out_channels + c) * out_depth + d) * out_height + h) * out_width + w;
        if (bias != nullptr) {
            sum += bias[c];
        }
        output[out_idx] = sum;
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

    const int total_elements = N * out_channels * out_depth * out_height * out_width;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_warp_shuffle_kernel", ([&] {
        transposed_conv3d_warp_shuffle_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with warp shuffle optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}