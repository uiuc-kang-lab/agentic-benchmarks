#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one output pixel per block and uses shared memory for intra-block reduction.
// Each thread computes partial sums over a subset of kernel elements, and then the partial sums are reduced
// using shared memory and warp-level primitives (__shfl_down_sync) for the final stage.

__global__ void depthwise_conv2d_kernel(
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
    int channels_per_group
) {
    // Each block computes one output pixel
    int out_index = blockIdx.x;
    if (out_index >= batch_size * out_channels * out_h * out_w) return;

    // Decode the output index into b, c, h, w
    int w_out = out_index % out_w;
    int rem = out_index / out_w;
    int h_out = rem % out_h;
    rem /= out_h;
    int c_out = rem % out_channels;
    int b = rem / out_channels;

    int g = c_out / channels_per_group;
    int m = c_out % channels_per_group; // for clarity

    int kernel_size = kernel_h * kernel_w;
    int tid = threadIdx.x;
    float partial_sum = 0.0f;

    // Each thread processes a subset of kernel elements
    for (int k = tid; k < kernel_size; k += blockDim.x) {
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int h_in = h_out * stride_h - padding_h + kh * dilation_h;
        int w_in = w_out * stride_w - padding_w + kw * dilation_w;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
            int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Reduction using shared memory
    extern __shared__ float sdata[];
    sdata[tid] = partial_sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction without __syncthreads
    if (tid < 32) {
        float sum_val = sdata[tid];
        // Perform warp-level reduction using shfl_down_sync
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            if (bias != nullptr) {
                sum_val += bias[c_out];
            }
            output[out_index] = sum_val;
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

    int total_outputs = batch_size * out_channels * out_h * out_w;
    int threads = 64; // Use 64 threads per block for reduction
    int blocks = total_outputs;
    size_t shared_mem = threads * sizeof(float);

    depthwise_conv2d_kernel<<<blocks, threads, shared_mem>>>(
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
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise Conv2D forward with optimized shared memory reduction (CUDA)");
}
