#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel assigns one warp to compute one output element of the transposed convolution.
// Each thread in the warp computes a partial sum over a subset of the input channels in the corresponding group
// and then a warp-level reduction using __shfl_down_sync aggregates the partial sums.


template <typename scalar_t>
__global__ void conv_transpose2d_warp_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // may be nullptr
    scalar_t* __restrict__ output,

    // Input dimensions
    int N, int C_in, int H_in, int W_in,
    // Output dimensions
    int C_out, int H_out, int W_out,
    // Kernel dimensions
    int kernel_h, int kernel_w,
    // Convolution parameters
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups
) {
    // Each warp computes one output element
    // Calculate global thread id
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpSize = 32;
    int warp_id = global_tid / warpSize;
    int lane = global_tid % warpSize;

    // Total number of output elements
    int total_outputs = N * C_out * H_out * W_out;
    if (warp_id >= total_outputs) return;

    // Decode the warp_id into (n, c_out, h_out, w_out)
    int tmp = warp_id;
    int w_out_idx = tmp % W_out;
    tmp /= W_out;
    int h_out_idx = tmp % H_out;
    tmp /= H_out;
    int c_out_idx = tmp % C_out;
    int n = tmp / C_out;

    // Determine group and channel partitioning
    int out_channels_per_group = C_out / groups;
    int in_channels_per_group = C_in / groups;
    int group = c_out_idx / out_channels_per_group;

    // Each warp will sum contributions from all kernel positions and input channels in the group
    // Partial sum computed per thread
    scalar_t partial_sum = 0;

    // Loop over kernel window positions
    for (int k_y = 0; k_y < kernel_h; ++k_y) {
      for (int k_x = 0; k_x < kernel_w; ++k_x) {
        // Compute the corresponding input spatial location
        int tmp_y = h_out_idx + pad_h - k_y * dilation_h;
        int tmp_x = w_out_idx + pad_w - k_x * dilation_w;
        if ((tmp_y % stride_h) != 0 || (tmp_x % stride_w) != 0) continue;
        int h_in = tmp_y / stride_h;
        int w_in = tmp_x / stride_w;
        if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;

        // Loop over input channels in the current group in a strided manner among warp lanes
        for (int c = lane; c < in_channels_per_group; c += warpSize) {
          int c_in = group * in_channels_per_group + c;
          int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
          // Weight layout: [C_in, out_channels_per_group, kernel_h, kernel_w]
          // Relative output channel index
          int rel_out = c_out_idx - group * out_channels_per_group;
          int weight_idx = (((c_in) * out_channels_per_group + rel_out) * kernel_h + k_y) * kernel_w + k_x;
          partial_sum += input[input_idx] * weight[weight_idx];
        }
      }
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Lane 0 writes the final result (adding bias if provided)
    if (lane == 0) {
      scalar_t sum = partial_sum;
      if (bias != nullptr) {
        sum += bias[c_out_idx];
      }
      int output_idx = ((n * C_out + c_out_idx) * H_out + h_out_idx) * W_out + w_out_idx;
      output[output_idx] = sum;
    }
}


// Host function to setup and launch the kernel

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Input dimensions: [N, C_in, H_in, W_in]
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Weight dimensions: [C_in, out_channels_per_group, kernel_h, kernel_w]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int C_out = out_channels_per_group * groups;

    int stride_h = stride[0];
    int stride_w = stride[1];
    int pad_h = padding[0];
    int pad_w = padding[1];
    int output_pad_h = output_padding[0];
    int output_pad_w = output_padding[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    // Calculate output dimensions using standard transposed convolution formula
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Total number of output elements that need to be computed by warps
    int total_outputs = N * C_out * H_out * W_out;
    const int warpSize = 32;
    // Each warp computes one output element, so total threads = total_outputs * warpSize
    int total_threads = total_outputs * warpSize;
    const int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_warp_kernel<scalar_t><<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cuda, "ConvTranspose2D forward with warp-level reduction (CUDA)");
}
