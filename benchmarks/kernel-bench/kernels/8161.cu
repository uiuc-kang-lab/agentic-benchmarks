#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_2d_blocked_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    // Use 2D thread blocks (16x16)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output position
    const int oh = by * blockDim.y + ty;
    const int ow = bx * blockDim.x + tx;

    // Early exit if outside output bounds
    if (oh >= out_height || ow >= out_width) return;

    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    // Process multiple channels per thread for better ILP
    #pragma unroll 4
    for (int n = 0; n < batch_size; n++) {
        if (n != bz) continue;

        for (int oc_block = 0; oc_block < out_channels; oc_block += 32) {
            #pragma unroll 4
            for (int oc_offset = 0; oc_offset < 32 && oc_block + oc_offset < out_channels; oc_offset++) {
                const int oc = oc_block + oc_offset;
                const int g = oc / out_channels_per_group;
                const int oc_group = oc % out_channels_per_group;

                scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

                const int ic_start = g * in_channels_per_group;
                
                // Compute input indices based on output position
                for (int kh = 0; kh < kernel_h; kh++) {
                    const int h_offset = oh + padding - kh * dilation;
                    if (h_offset < 0 || h_offset % stride != 0) continue;
                    const int h_in = h_offset / stride;
                    if (h_in >= in_height) continue;

                    for (int kw = 0; kw < kernel_w; kw++) {
                        const int w_offset = ow + padding - kw * dilation;
                        if (w_offset < 0 || w_offset % stride != 0) continue;
                        const int w_in = w_offset / stride;
                        if (w_in >= in_width) continue;

                        #pragma unroll 4
                        for (int ic = 0; ic < in_channels_per_group; ic++) {
                            const int input_idx = n * (in_channels * in_height * in_width) +
                                                (ic_start + ic) * (in_height * in_width) +
                                                h_in * in_width + w_in;

                            const int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                 oc_group * (kernel_h * kernel_w) +
                                                 kh * kernel_w + kw;

                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }

                // Write output
                const int out_idx = n * (out_channels * out_height * out_width) +
                                  oc * (out_height * out_width) +
                                  oh * out_width + ow;
                output[out_idx] = sum;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Use 2D thread blocks of size 16x16
    dim3 threads(16, 16);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_2d_blocked_cuda", ([&] {
        conv_transpose2d_2d_blocked_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Blocked Transposed Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}