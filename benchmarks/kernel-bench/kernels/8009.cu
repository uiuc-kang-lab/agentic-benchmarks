#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// Macros to check CUDA tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Custom CUDA kernel for transposed 1D convolution using warp-level reduction
// Each warp (32 threads) computes one output element by iterating over a portion of the summation domain
// and then reducing the partial sums with __shfl_down_sync. This eliminates shared memory usage for reductions.
__global__ void conv_transposed_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_width,
    int groups) {

    // Determine the output element this warp is responsible for
    // Linear index mapping: output index = b * (out_channels * output_width) + out_c * output_width + pos
    int linear_idx = blockIdx.x;  // one block per output element
    int pos = linear_idx % output_width;
    linear_idx /= output_width;
    int out_c = linear_idx % out_channels;
    int b = linear_idx / out_channels;

    float sum = 0.0f;
    int lane = threadIdx.x; // Lane index within the warp (assumed blockDim.x == 32)

    // Determine channel range based on groups
    int c_start = 0, c_end = in_channels;
    int out_channels_per_group = out_channels; // when groups == 1
    if (groups > 1) {
        int group_in_channels = in_channels / groups;
        out_channels_per_group = out_channels / groups;
        int group_id = out_c / out_channels_per_group;
        c_start = group_id * group_in_channels;
        c_end = c_start + group_in_channels;
    }

    // The total number of (channel, input position) pairs to reduce
    int total = (c_end - c_start) * input_width;
    
    // Each warp thread processes a subset of the (channel, j) pairs.
    // We flatten the 2D loop: for each channel c in [c_start, c_end) and for each input index j in [0, input_width)
    // Compute the corresponding kernel index k = pos + padding - j*stride, and if valid, accumulate the product.
    for (int idx = lane; idx < total; idx += 32) {
        int channel_offset = idx / input_width;  // offset within the channel range
        int j = idx % input_width;
        int c = c_start + channel_offset;

        int k = pos + padding - j * stride; // derived from: pos = j*stride - padding + k
        if (k >= 0 && k < kernel_size) {
            // Access input: shape [batch, in_channels, input_width]
            float x_val = input[b * in_channels * input_width + c * input_width + j];
            float w_val;
            if (groups == 1) {
                // weight shape: [in_channels, out_channels, kernel_size]
                w_val = weight[c * out_channels * kernel_size + out_c * kernel_size + k];
            } else {
                // For groups > 1, weight shape: [in_channels, out_channels_per_group, kernel_size]
                int group_id = out_c / out_channels_per_group;
                int weight_c = c - group_id * (in_channels / groups);
                int out_c_in_group = out_c - group_id * out_channels_per_group;
                w_val = weight[weight_c * (out_channels_per_group * kernel_size) + out_c_in_group * kernel_size + k];
            }
            sum += x_val * w_val;
        }
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    
    // Write the result from lane 0 of the warp to global memory
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        output[b * out_channels * output_width + out_c * output_width + pos] = sum;
    }
}


// Forward function that sets up and calls the custom CUDA kernel
// It computes the output width based on input dimensions and convolution parameters

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups >= 1, "groups must be >= 1");

    // x shape: [batch, in_channels, input_width]
    auto batch = x.size(0);
    auto in_channels = x.size(1);
    auto input_width = x.size(2);
    int kernel_size = weight.size(2);

    int out_channels;
    if (groups == 1) {
        out_channels = weight.size(1);
    } else {
        out_channels = groups * weight.size(1); // weight shape: [in_channels, out_channels_per_group, kernel_size]
    }

    // Compute output width for conv_transpose1d
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = x.options();
    auto output = torch::zeros({batch, out_channels, output_width}, options);

    // Total number of output elements: each is computed by one warp
    int total_outputs = batch * out_channels * output_width;
    const int threads = 32;  // one warp per output element

    // Launch our custom kernel on the current CUDA stream
    conv_transposed_1d_kernel<<<total_outputs, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        input_width,
        kernel_size,
        stride,
        padding,
        output_width,
        groups
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with warp-level reduction");
}
