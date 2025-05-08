#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction kernel for 3D convolution
// Each warp collaboratively computes one output element using __shfl_down_sync

template <typename scalar_t>
__global__ void conv3d_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,  // can be nullptr if not provided
    int batch_size,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding,
    int dilation,
    int groups,
    int out_channels_per_group) {

    // Global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Lane index within a warp (0-31)
    int lane = tid & 31;
    // Global warp id
    int warp_id_global = tid >> 5;
    // Total warps in the grid
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_d * out_h * out_w;

    // Loop over output elements on a warp-granularity basis
    for (int out_idx = warp_id_global; out_idx < total_outputs; out_idx += total_warps) {
        // Decode linear index into 5D indices: [b, oc, od, oh, ow]
        int tmp = out_idx;
        int ow = tmp % out_w; tmp /= out_w;
        int oh = tmp % out_h; tmp /= out_h;
        int od = tmp % out_d; tmp /= out_d;
        int oc = tmp % out_channels; tmp /= out_channels;
        int b  = tmp;

        // For the given output channel, determine the input channel group
        int in_channels_per_group = in_channels / groups;
        int group = oc / out_channels_per_group;
        int input_channel_base = group * in_channels_per_group;

        // Total number of multiplications for one output element
        int num_iters = in_channels_per_group * kernel_d * kernel_h * kernel_w;
        scalar_t partial_sum = static_cast<scalar_t>(0);

        // Distribute the reduction work among warp lanes
        for (int i = lane; i < num_iters; i += 32) {
            // Compute indices for input channel and kernel positions
            int ic = i / (kernel_d * kernel_h * kernel_w);
            int rem = i % (kernel_d * kernel_h * kernel_w);
            int kd = rem / (kernel_h * kernel_w);
            rem = rem % (kernel_h * kernel_w);
            int kh = rem / kernel_w;
            int kw = rem % kernel_w;
            
            int in_channel = input_channel_base + ic;
            
            // Compute the corresponding input indices
            int id = od * stride - padding + kd * dilation;
            int ih = oh * stride - padding + kh * dilation;
            int iw = ow * stride - padding + kw * dilation;
            
            // Check for valid input bounds
            if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_index = (((b * in_channels + in_channel) * in_d + id) * in_h + ih) * in_w + iw;
                int weight_index = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                partial_sum += input[input_index] * weight[weight_index];
            }
        }

        // Perform warp-level reduction using __shfl_down_sync
        // Full mask for 32 threads
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        // Lane 0 writes the result for this output element
        if (lane == 0) {
            if (bias != nullptr) {
                partial_sum += bias[oc];
            }
            output[out_idx] = partial_sum;
        }
    }
}

// Host forward function
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

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

    // Compute output dimensions using standard convolution formulas
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, input.options());

    int total_outputs = batch_size * out_channels * out_d * out_h * out_w;
    // Each warp (32 threads) computes one output element. Compute number of warps needed.
    int num_warps_needed = (total_outputs + 31) / 32;
    // Each block will have 256 threads (8 warps per block)
    int threads = 256;
    int blocks = (num_warps_needed + 7) / 8;

    int out_channels_per_group = out_channels / groups;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_warp_reduce_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        const scalar_t* bias_ptr = nullptr;
        if (bias.defined()) {
            bias_ptr = bias.data_ptr<scalar_t>();
        }
        conv3d_warp_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input_ptr,
            weight_ptr,
            output_ptr,
            bias_ptr,
            batch_size, in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups, out_channels_per_group);
        cudaDeviceSynchronize();
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using warp-level reduction (CUDA)");
}
