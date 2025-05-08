#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 128
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = batch * channels * out_h * out_w;
    
    if (idx >= total) return;

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int c = (idx / (out_w * out_h)) % channels;
    const int n = idx / (out_w * out_h * channels);

    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        #pragma unroll
        for (int j = 0; j < k; ++j) {
            const int ih = oh * stride - padding + i * dilation;
            const int iw = ow * stride - padding + j * dilation;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                sum += input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] *
                       weight[c * k * k + i * k + j];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c];
    }
    output[idx] = sum;
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {
    
    // Warp-based reduction for channel processing
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int out_x = blockIdx.x * blockDim.x + warp_id;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_ch = blockIdx.z % out_channels;
    const int batch_idx = blockIdx.z / out_channels;

    if (out_x >= width || out_y >= height || batch_idx >= batch) return;

    const int spatial_offset = out_y * width + out_x;
    const int weight_offset = out_ch * in_channels;
    const int input_batch_offset = batch_idx * in_channels * height * width;

    scalar_t sum = 0;
    
    // Warp-stride loop with vectorized reduction
    for (int in_ch = lane_id; in_ch < in_channels; in_ch += WARP_SIZE) {
        sum += input[input_batch_offset + in_ch * height * width + spatial_offset] *
               weight[weight_offset + in_ch];
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First lane writes final result
    if (lane_id == 0) {
        scalar_t result = sum + (bias ? bias[out_ch] : 0);
        output[batch_idx * out_channels * height * width +
               out_ch * height * width +
               spatial_offset] = result;
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

    const int total_threads_dw = batch * in_channels * out_h * out_w;
    const int blocks_dw = (total_threads_dw + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_dw, BLOCK_SIZE>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));

    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    // Optimized pointwise configuration
    dim3 threads(WARP_SIZE * 2, 2);  // 64 threads per block
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch * out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
    }));

    return output;
}

// Wrapper function for Python binding
torch::Tensor forward_wrapper(
    const torch::Tensor& input,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {
    
    return forward_cuda(
        input, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias,
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}