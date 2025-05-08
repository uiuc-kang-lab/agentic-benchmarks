#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inline clamp function (assumed to compile to branchless code on CUDA)
__device__ inline int clamp_int(int x, int lo, int hi) {
    // Use standard ternary operators; on CUDA these are usually compiled efficiently
    return x < lo ? lo : (x > hi ? hi : x);
}

// Custom 3D convolution kernel optimized to minimize warp divergence by using branchless
// arithmetic for boundary condition handling. This kernel supports an optional bias and groups
// (assumes groups divide the channel dimensions evenly). The kernel uses clamped index reads
// with a branchless validity mask so that out-of-bound positions are read from a valid location
// but their contribution is zero.
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups,
    int with_bias  // 1 if bias is defined, 0 otherwise
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_depth * out_height * out_width;
    if (index >= total) return;

    // Decode the flattened index into [n, oc, od, oh, ow]
    int ow_idx = index % out_width;
    int tmp = index / out_width;
    int oh_idx = tmp % out_height;
    tmp = tmp / out_height;
    int od_idx = tmp % out_depth;
    tmp = tmp / out_depth;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;

    // Determine the range of input channels for this output channel using groups
    int group_size = in_channels / groups;
    int out_channels_per_group = out_channels / groups; // also assumed to be even
    int group = oc / out_channels_per_group;
    int ic_start = group * group_size;
    int ic_end = ic_start + group_size;

    // Loop over the input channels (within the appropriate group) and kernel volumes
    for (int ic = ic_start; ic < ic_end; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            // Compute input depth index
            int d_in = od_idx * stride - padding + kd * dilation;
            // Compute validity flag branchlessly
            int valid_d = ((unsigned)d_in < (unsigned)in_depth) ? 1 : 0;
            // Clamp d_in to a legal index (this read is safe even if out-of-bound, since its contribution
            // is subsequently multiplied by the 0 validity flag)
            int d_clamped = clamp_int(d_in, 0, in_depth - 1);
            
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in = oh_idx * stride - padding + kh * dilation;
                int valid_h = ((unsigned)h_in < (unsigned)in_height) ? 1 : 0;
                int h_clamped = clamp_int(h_in, 0, in_height - 1);
                
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in = ow_idx * stride - padding + kw * dilation;
                    int valid_w = ((unsigned)w_in < (unsigned)in_width) ? 1 : 0;
                    int w_clamped = clamp_int(w_in, 0, in_width - 1);

                    // Combine validity from each spatial dimension (branchlessly)
                    int valid = valid_d * valid_h * valid_w;

                    // Compute flattened index for input tensor [n, ic, d, h, w]
                    int input_index = n * (in_channels * in_depth * in_height * in_width)
                                      + ic * (in_depth * in_height * in_width)
                                      + d_clamped * (in_height * in_width)
                                      + h_clamped * in_width
                                      + w_clamped;

                    // Weight tensor shape: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
                    int relative_ic = ic - ic_start;
                    int weight_index = oc * (group_size * kernel_d * kernel_h * kernel_w)
                                       + relative_ic * (kernel_d * kernel_h * kernel_w)
                                       + kd * (kernel_h * kernel_w)
                                       + kh * kernel_w
                                       + kw;

                    sum += valid * input[input_index] * weight[weight_index];
                }
            }
        }
    }

    if (with_bias) {
        sum += bias[oc];
    }

    // Write the computed sum to the flattened output
    output[index] = sum;
}

// Host function that wraps the CUDA kernel launch
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    at::Tensor bias;
    bool bias_defined = false;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        bias_defined = true;
    }

    // Input shape: [N, C, D, H, W]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Weight shape: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output spatial dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Launch kernel with one thread per output element
    int total = batch_size * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv3d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_defined ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        dilation,
        groups,
        bias_defined ? 1 : 0
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward optimized with branchless control flow (CUDA)");
}
