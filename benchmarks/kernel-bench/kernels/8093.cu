#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Each thread computes one output element in [batch, out_channel, out_width]
// Memory coalescing is ensured for the output write as threads in a block access consecutive out_width indices.

template <typename scalar_t>
__global__ void coalesced_conv_transposed_1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr if not provided
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_width,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    // Determine output index
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;  // each block y corresponds to an output channel
    int n  = blockIdx.z;  // each block z corresponds to a sample in the batch

    if (ow >= out_width) return;

    // Determine group parameters
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group  = in_channels / groups;
    int group = oc / out_channels_per_group;
    int oc_in_group = oc - group * out_channels_per_group;

    // Initialize with bias if provided
    scalar_t value = 0;
    if (bias != nullptr) {
        value = bias[oc];
    }

    // Precompute offset for current output width index to help coalesced access and reduce recomputation
    int offset = ow + padding;

    // For each input channel in the appropriate group
    int ic_start = group * in_channels_per_group;
    int ic_end = ic_start + in_channels_per_group;
    for (int ic = ic_start; ic < ic_end; ic++) {
        // Iterate over kernel positions
        for (int k = 0; k < kernel_size; k++) {
            int diff = offset - k;
            if (diff < 0) break;  // as k increases, diff will only decrease
            if (diff % stride == 0) {
                int iw = diff / stride;
                if (iw < in_width) {
                    int input_idx = n * (in_channels * in_width) + ic * in_width + iw;
                    int weight_idx = ic * (out_channels_per_group * kernel_size) + (oc_in_group * kernel_size) + k;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Write the computed output element
    int output_idx = n * (out_channels * out_width) + oc * out_width + ow;
    output[output_idx] = value;
}

// Host function

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // Dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_width = input.size(2);
    int kernel_size = weight.size(2);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output_tensor = torch::empty({batch_size, out_channels, out_width}, input.options());

    // Launch parameters: each thread computes one output element along the out_width dimension
    const int threads = 256;
    const int blocks_x = (out_width + threads - 1) / threads;
    dim3 block(threads);
    dim3 grid(blocks_x, out_channels, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "coalesced_conv_transposed_1d_forward", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        const scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
        const scalar_t* bias_ptr = (bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr);
        scalar_t* output_ptr = output_tensor.data_ptr<scalar_t>();

        coalesced_conv_transposed_1d_kernel<scalar_t><<<grid, block>>>(
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch_size,
            in_channels,
            out_channels,
            in_width,
            out_width,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups);
    }));

    cudaDeviceSynchronize();
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Transposed 1D Convolution forward (CUDA)");
}
