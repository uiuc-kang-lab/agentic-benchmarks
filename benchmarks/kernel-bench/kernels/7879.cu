#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define PART_SIZE 16
#define BLOCK_SIZE 256

// Kernel that partitions the input channel dimension among threads and uses atomicAdd
// to accumulate partial convolution results. Each thread computes the partial sum for a
// single output element over a slice of the input channels, then atomically adds it to
// the final output. Atomic operations are used only at the final accumulation step,
// minimizing global memory contention.

__global__ void conv2d_atomic_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int out_h,
    const int out_w) {

    // Total number of output elements
    int total_out = batch_size * out_channels * out_h * out_w;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    // Decode output index into (b, oc, oh, ow) assuming contiguous layout [b, oc, out_h, out_w]
    int temp = out_idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int oc = temp % out_channels; temp /= out_channels;
    int b = temp;  

    // Determine the valid range for kernel iteration based on padding and stride
    int kh_start = max(0, padding - oh * stride);
    int kh_end   = min(kernel_h, height - oh * stride + padding);
    int kw_start = max(0, padding - ow * stride);
    int kw_end   = min(kernel_w, width - ow * stride + padding);

    // Partition the input channels along the channel dimension
    int part = blockIdx.y;  // each y-dimension block handles a slice of input channels
    int start_ic = part * PART_SIZE;
    int end_ic = start_ic + PART_SIZE;
    if (end_ic > in_channels) end_ic = in_channels;

    float partial_sum = 0.0f;

    // Loop over the assigned input channels and valid kernel window
    for (int ic = start_ic; ic < end_ic; ic++) {
        for (int kh = kh_start; kh < kh_end; kh++) {
            int h_in = oh * stride + kh - padding;
            for (int kw = kw_start; kw < kw_end; kw++) {
                int w_in = ow * stride + kw - padding;
                float in_val = __ldg(&input[((b * in_channels + ic) * height + h_in) * width + w_in]);
                float w_val = __ldg(&weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw]);
                partial_sum += in_val * w_val;
            }
        }
    }
    
    // Atomically accumulate the partial result into the proper output element
    atomicAdd(&output[((b * out_channels + oc) * out_h + oh) * out_w + ow], partial_sum);
}


torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    if (dilation != 1 || groups != 1) {
        return torch::conv2d(x, weight, bias,
                             {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int out_h = (height + 2 * padding - kernel_h) / stride + 1;
    int out_w = (width + 2 * padding - kernel_w) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    int total_out = batch_size * out_channels * out_h * out_w;
    // Split the input channels into partitions; each partition will sum over PART_SIZE channels
    int num_parts = (in_channels + PART_SIZE - 1) / PART_SIZE;

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((total_out + BLOCK_SIZE - 1) / BLOCK_SIZE, num_parts);

    conv2d_atomic_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride,
        padding,
        out_h,
        out_w
    );

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with atomic reduction");
}
