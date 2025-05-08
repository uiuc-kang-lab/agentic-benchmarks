#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Modular device function to compute the convolution sum for one output element
template <typename scalar_t>
__device__ inline scalar_t compute_conv3d(
    const scalar_t* input,
    const scalar_t* weight,
    int in_channels, int in_d, int in_h, int in_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int out_channels_per_group,
    int b, int oc, int od, int oh, int ow) {

    scalar_t sum = 0;
    int in_channels_per_group = in_channels / groups;
    // Determine which group this output channel belongs to
    int group = oc / out_channels_per_group;
    // Weight tensor is stored in [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int weight_offset = oc * in_channels_per_group * kernel_d * kernel_h * kernel_w;
    int input_channel_offset = group * in_channels_per_group;

    // Loop over the channels in this group
    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        int input_channel = input_channel_offset + ic;
        // Iterate over kernel depth
        for (int kd = 0; kd < kernel_d; ++kd) {
            int id = od * stride - padding + kd * dilation;
            if (id < 0 || id >= in_d) continue;
            // Iterate over kernel height
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = oh * stride - padding + kh * dilation;
                if (ih < 0 || ih >= in_h) continue;
                // Iterate over kernel width
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = ow * stride - padding + kw * dilation;
                    if (iw < 0 || iw >= in_w) continue;
                    // Compute indices into the input and weight tensors
                    int input_idx = (((b * in_channels + input_channel) * in_d + id) * in_h + ih) * in_w + iw;
                    int weight_idx = weight_offset + (((ic * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    return sum;
}

// Main convolution kernel that invokes the modular device function
template <typename scalar_t>
__global__ void conv3d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size, int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups, int out_channels_per_group) {

    int total = batch_size * out_channels * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < total) {
        // Decompose linear index into 5D indices [b, oc, od, oh, ow]
        int w = idx % out_w;
        int temp = idx / out_w;
        int h = temp % out_h;
        temp /= out_h;
        int d = temp % out_d;
        temp /= out_d;
        int oc = temp % out_channels;
        int b = temp / out_channels;

        scalar_t value = compute_conv3d<scalar_t>(
            input, weight,
            in_channels, in_d, in_h, in_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, out_channels_per_group,
            b, oc, d, h, w);

        output[idx] = value;
        idx += blockDim.x * gridDim.x;
    }
}

// Optional bias addition kernel as a separate modular function
template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t* output,
    const scalar_t* bias,
    int batch_size, int out_channels, int out_d, int out_h, int out_w) {

    int total = batch_size * out_channels * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < total) {
        int w = idx % out_w;
        int temp = idx / out_w;
        int h = temp % out_h;
        temp /= out_h;
        int d = temp % out_d;
        temp /= out_d;
        int oc = temp % out_channels;
        
        // Bias is stored per output channel
        output[idx] += bias[oc];
        idx += blockDim.x * gridDim.x;
    }
}

// Host forward function that sets up convolution parameters and launches the kernels
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }
    
    // Input dimensions: [batch, in_channels, in_d, in_h, in_w]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Weight dimensions: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Calculate output dimensions using standard convolution formula
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    int out_channels_per_group = out_channels / groups;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        conv3d_kernel<scalar_t><<<blocks, threads>>>(
            input_ptr, weight_ptr, output_ptr,
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation, groups, out_channels_per_group);
        cudaDeviceSynchronize();

        if (bias.defined()) {
            const auto* bias_ptr = bias.data_ptr<scalar_t>();
            add_bias_kernel<scalar_t><<<blocks, threads>>>(
                output_ptr, bias_ptr,
                batch_size, out_channels, out_d, out_h, out_w);
            cudaDeviceSynchronize();
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular 3D convolution forward CUDA kernel");
}
