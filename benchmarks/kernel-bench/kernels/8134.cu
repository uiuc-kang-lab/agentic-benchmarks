#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel: each block computes one output pixel by parallelizing the inner reduction
// over the input channel and kernel dimensions using shared memory and warp-level primitives.

template <typename scalar_t>
__global__ void conv_transpose2d_kernel_opt_shared_reduction(
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
    // Each block computes one output element
    int out_idx = blockIdx.x;
    int tmp = out_idx;
    int ow = tmp % out_width; tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;  // batch index

    // Compute group info
    int out_channels_per_group = out_channels / groups;
    int g = oc / out_channels_per_group;
    int oc_group = oc % out_channels_per_group;
    int in_channels_per_group = in_channels / groups;
    int ic_start = g * in_channels_per_group;

    // Total reduction size: over kernel (kh,kw) and input channel within the group
    int red_size = in_channels_per_group * kernel_h * kernel_w;
    scalar_t sum = 0;

    // Each thread in the block accumulates a portion of the reduction
    for (int r = threadIdx.x; r < red_size; r += blockDim.x) {
        int ic = r / (kernel_h * kernel_w);
        int rem = r % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        int h_in_tmp = oh - kh * dilation + padding;
        int w_in_tmp = ow - kw * dilation + padding;
        if (h_in_tmp < 0 || w_in_tmp < 0) continue;
        if ((h_in_tmp % stride) != 0 || (w_in_tmp % stride) != 0) continue;
        int h_in = h_in_tmp / stride;
        int w_in = w_in_tmp / stride;
        if (h_in < 0 || h_in >= in_height || w_in < 0 || w_in >= in_width) continue;

        int input_index = b * (in_channels * in_height * in_width) +
                          (ic_start + ic) * (in_height * in_width) +
                          h_in * in_width + w_in;
        int weight_index = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                           oc_group * (kernel_h * kernel_w) +
                           kh * kernel_w + kw;
        sum += input[input_index] * weight[weight_index];
    }

    // Reduction across threads in this block using shared memory and warp-level primitives
    extern __shared__ scalar_t sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce to 32 threads using shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        // Use warp-level reduction with __shfl_down_sync
        scalar_t val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        sdata[threadIdx.x] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t finalVal = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
        output[out_idx] = finalVal + sdata[0];
    }
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    
    // Calculate output channels as: weight shape: [in_channels, out_channels/groups, kH, kW]
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
    }

    // Compute output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    // Total number of output elements
    const int total_output = output.numel();

    // Choose a reasonable block size (must be power of 2); here we use 256 threads per block.
    constexpr int BLOCK_SIZE = 256;
    dim3 blocks(total_output);
    dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda_opt_shared_reduction", ([&] {
        conv_transpose2d_kernel_opt_shared_reduction<scalar_t><<<blocks, threads, BLOCK_SIZE * sizeof(scalar_t)>>>(
            x.data_ptr<scalar_t>(),
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
    m.def("forward", &forward, "Optimized Transposed 2D Convolution with Shared Memory Reduction (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
