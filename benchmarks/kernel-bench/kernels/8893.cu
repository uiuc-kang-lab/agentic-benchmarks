#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__device__ __forceinline__ int gcd(int a, int b) {
    while(b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

__global__ void conv_transpose2d_kernel_warp_aligned(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    // Calculate global thread index and warp index within block
    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    // Shared memory for bias values - aligned to warp size
    __shared__ float s_bias[BLOCK_SIZE];
    
    // Pre-load bias into shared memory (warp-aligned access)
    if (tidx < out_channels) {
        s_bias[threadIdx.x] = bias[tidx];
    }
    __syncthreads();

    // Calculate total output elements and stride for thread processing
    const int total_outputs = batch * out_channels * out_h * out_w;
    const int thread_stride = gridDim.x * blockDim.x;
    
    // Pre-compute stride and dilation related values
    const int stride_h_gcd = gcd(stride_h, dilation_h);
    const int stride_w_gcd = gcd(stride_w, dilation_w);
    const int step_h = stride_h / stride_h_gcd;
    const int step_w = stride_w / stride_w_gcd;

    // Process multiple output elements per thread
    for (int idx = tidx; idx < total_outputs; idx += thread_stride) {
        // Decode output indices
        const int ow = idx % out_w;
        int tmp = idx / out_w;
        const int oh = tmp % out_h;
        tmp = tmp / out_h;
        const int oc = tmp % out_channels;
        const int n = tmp / out_channels;

        // Initialize output with bias
        float out_val = s_bias[oc % BLOCK_SIZE];

        // Calculate group and input channel range
        const int g = oc / out_channels_per_group;
        const int ic_start = g * in_channels_per_group;
        const int ic_end = (g + 1) * in_channels_per_group;

        // Pre-compute spatial offsets
        const int h_off = oh + pad_h;
        const int w_off = ow + pad_w;

        // Pre-compute kernel bounds
        const int kh_start = (h_off % stride_h + stride_h) % stride_h;
        const int kw_start = (w_off % stride_w + stride_w) % stride_w;
        const int kh_end = min(kernel_h, (h_off / dilation_h) + 1);
        const int kw_end = min(kernel_w, (w_off / dilation_w) + 1);

        // Main computation loops with minimal branching
        #pragma unroll 4
        for (int kh = kh_start; kh < kh_end; kh += step_h) {
            const int ih = (h_off - kh * dilation_h) / stride_h;
            const bool valid_h = ih >= 0 && ih < in_h;

            #pragma unroll 4
            for (int kw = kw_start; kw < kw_end; kw += step_w) {
                const int iw = (w_off - kw * dilation_w) / stride_w;
                const bool valid_w = iw >= 0 && iw < in_w;

                if (valid_h && valid_w) {
                    #pragma unroll 4
                    for (int ic = ic_start; ic < ic_end; ic++) {
                        const int x_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                        const int w_idx = ((ic * out_channels_per_group + 
                                        (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                        
                        out_val += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
                    }
                }
            }
        }

        // Write output
        output[idx] = out_val;
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.has_value() && bias.value().defined())
        bias = bias.value().contiguous();

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;

    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({out_channels}, weight.options());
    }

    auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

    // Calculate grid dimensions based on warp-aligned blocks
    const int total_elements = batch * out_channels * out_h * out_w;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_dim = min(num_blocks, 65535); // Respect maximum grid dimension

    conv_transpose2d_kernel_warp_aligned<<<grid_dim, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
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
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels / groups,
        out_channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned 2D Transposed Convolution (CUDA)");
}