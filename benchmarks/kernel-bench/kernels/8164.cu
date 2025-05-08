#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel uses shared memory for reduction and warp-level primitives for final stages.
template <typename scalar_t>
__global__ void conv_transpose2d_shared_reduce_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias, // may be nullptr
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
    extern __shared__ scalar_t shared_bf_reduce[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    int tx = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;

    // Unravel the flat index to (b, oc, oh, ow)
    int tmp = idx;
    int ow = tmp % out_width;
    tmp /= out_width;
    int oh = tmp % out_height;
    tmp /= out_height;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Determine group-related indices
    int out_channels_per_group = out_channels / groups;
    int g = oc / out_channels_per_group;
    int oc_group = oc % out_channels_per_group;
    int in_channels_per_group = in_channels / groups;
    int in_channel_start = g * in_channels_per_group;

    scalar_t value = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);

    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_offset = oh + padding - kh * dilation;
        if (h_offset < 0 || h_offset % stride != 0) continue;
        int h_in = h_offset / stride;
        if (h_in < 0 || h_in >= in_height) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_offset = ow + padding - kw * dilation;
            if (w_offset < 0 || w_offset % stride != 0) continue;
            int w_in = w_offset / stride;
            if (w_in < 0 || w_in >= in_width) continue;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int input_channel = in_channel_start + ic;
                int input_idx = b * (in_channels * in_height * in_width) 
                              + input_channel * (in_height * in_width) 
                              + h_in * in_width + w_in;
                scalar_t in_val = input[input_idx];

                int weight_idx = input_channel * (out_channels_per_group * kernel_h * kernel_w) 
                               + oc_group * (kernel_h * kernel_w) 
                               + kh * kernel_w + kw;
                scalar_t weight_val = weight[weight_idx];

                value += in_val * weight_val;
            }
        }
    }

    shared_bf_reduce[tx] = value;
    __syncthreads();

    if (WARP_SIZE > 1) {
        if (tx < WARP_SIZE) {
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
        }
    }

    if (tx == 0) {
        int output_idx = b * (out_channels * out_height * out_width) 
                       + oc * (out_height * out_width) 
                       + oh * out_width + ow;
        atomicAdd(&output[output_idx], value);
    }
}

// Forward function: prepares output tensor and launches the shared memory reduce kernel.
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
    const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    size_t shared_mem_size = threads * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_shared_reduce_cuda", ([&] {
        const scalar_t* bias_data = (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr;
        conv_transpose2d_shared_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_data,
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
    m.def("forward", &forward, "Shared Memory Reduction Transposed 2D Convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"),
          py::arg("dilation") = 1);
}