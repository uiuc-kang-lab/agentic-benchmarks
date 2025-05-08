#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction for summing floats
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Kernel where each warp cooperatively computes one output element
__global__ void warp_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group,
    int total_output
) {
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpSizeLocal = 32; // assuming warp size is 32
    int warpId = tid / warpSizeLocal;
    int laneId = tid % warpSizeLocal;
    int numThreads = gridDim.x * blockDim.x;
    int numWarps = numThreads / warpSizeLocal;

    // Each warp processes one output pixel at a time
    for (int out_idx = warpId; out_idx < total_output; out_idx += numWarps) {
        // Decode the flat output index into (b, c_out, h_out, w_out)
        int tmp = out_idx;
        int w_out_idx = tmp % out_w;
        tmp /= out_w;
        int h_out_idx = tmp % out_h;
        tmp /= out_h;
        int c_out = tmp % out_channels;
        int b = tmp / out_channels;

        // For depthwise convolution, determine group index and intra-group channel index
        int g = c_out / channels_per_group;
        int m = c_out % channels_per_group;

        float sum = 0.0f;
        int kernel_size = kernel_h * kernel_w;

        // Each lane in the warp processes a subset of kernel elements
        for (int k = laneId; k < kernel_size; k += warpSizeLocal) {
            int kh = k / kernel_w;
            int kw = k % kernel_w;

            int h_in = h_out_idx * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out_idx * stride_w - padding_w + kw * dilation_w;

            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }

        // Perform warp-level reduction to accumulate partial sums
        sum = warpReduceSum(sum);

        // Lane 0 writes the result for this output pixel
        if (laneId == 0) {
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            int out_index = ((b * out_channels + c_out) * out_h + h_out_idx) * out_w + w_out_idx;
            output[out_index] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cuda(), "bias must be a CUDA tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    int total_output = batch_size * out_channels * out_h * out_w;
    
    // Determine grid and block sizes so that one warp computes one output pixel
    int threads_per_block = 256; // multiple of warp size
    // Total threads rounded up to cover all output pixels by warps
    int total_threads = ((total_output + 31) / 32) * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    warp_depthwise_conv2d_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group,
        total_output
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Reduction Depthwise Conv2D forward (CUDA)");
}
