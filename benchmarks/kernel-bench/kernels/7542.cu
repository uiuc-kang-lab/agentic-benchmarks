#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kT, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups,
    int weight_elements_per_filter
) {
    extern __shared__ scalar_t shared_weights[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_channels * out_depth * out_height * out_width) return;

    // Decompose output index
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
    scalar_t sum = 0;
    int in_channels_per_group = in_channels / groups;

    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int input_channel = group * in_channels_per_group + ic;

        // Load weights for current input channel to shared memory
        for (int i = threadIdx.x; i < weight_elements_per_filter; i += blockDim.x) {
            int oc_local = i / (kT * kH * kW);
            int k_off = i % (kT * kH * kW);
            int kd = k_off / (kH * kW);
            int kh = (k_off % (kH * kW)) / kW;
            int kw = k_off % kW;
            int weight_idx = (((input_channel * (out_channels / groups) + oc_local) * kT + kd) * kH + kh) * kW + kw;
            shared_weights[i] = weight[weight_idx];
        }
        __syncthreads();

        // Main computation with shared weights
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
                    int weight_shared_idx = out_c_local * kT*kH*kW + kd*kH*kW + kh*kW + kw;
                    sum += input[input_idx] * shared_weights[weight_shared_idx];
                }
            }
        }
        __syncthreads();
    }

    if (bias != nullptr) sum += bias[c];
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
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) bias_tensor = bias.value().contiguous();

    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);
    int out_channels = weight.size(1) * groups;

    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());
    int total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    int weight_elements_per_filter = (out_channels / groups) * kT * kH * kW;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_shared_kernel", ([&] {
        transposed_conv3d_shared_kernel<scalar_t><<<blocks, threads, weight_elements_per_filter * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            groups,
            weight_elements_per_filter
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3D with shared memory weights",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}