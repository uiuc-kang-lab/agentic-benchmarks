#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

__global__ void conv_transpose2d_warp_kernel(
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

    // Calculate output position
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    
    // Each warp processes one output element
    int total_warps = (blockDim.x * gridDim.x) / warp_size;
    int warp_index = blockIdx.x * (blockDim.x / warp_size) + warp_id;
    
    for (; warp_index < batch * out_channels * out_h * out_w; warp_index += total_warps) {
        // Decode position
        int ow = warp_index % out_w;
        int tmp = warp_index / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        float partial_sum = 0.0f;
        
        // Get group
        int g = oc / out_channels_per_group;
        int local_oc = oc - g * out_channels_per_group;

        // Each thread in warp handles different input channels
        for (int c = g * in_channels_per_group + lane_id; 
             c < (g + 1) * in_channels_per_group; 
             c += warp_size) {
            
            #pragma unroll 4
            for (int kh = 0; kh < kernel_h; kh++) {
                int h_in_candidate = oh + pad_h - kh * dilation_h;
                if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
                int ih = h_in_candidate / stride_h;
                if (ih >= in_h) continue;

                #pragma unroll 4
                for (int kw = 0; kw < kernel_w; kw++) {
                    int w_in_candidate = ow + pad_w - kw * dilation_w;
                    if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
                    int iw = w_in_candidate / stride_w;
                    if (iw >= in_w) continue;

                    int x_index = n * (in_channels * in_h * in_w) +
                                c * (in_h * in_w) +
                                ih * in_w + iw;

                    int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                                     local_oc * (kernel_h * kernel_w) +
                                     kh * kernel_w + kw;

                    partial_sum += x[x_index] * weight[weight_index];
                }
            }
        }

        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        // First thread in warp writes result
        if (lane_id == 0) {
            int out_index = n * (out_channels * out_h * out_w) +
                          oc * (out_h * out_w) +
                          oh * out_w + ow;
            output[out_index] = partial_sum + bias[oc];
        }
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

    int in_channels_per_group = in_channels / groups;

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int num_warps_per_block = threads_per_block / 32;
    const int num_blocks = (batch * out_channels * out_h * out_w + num_warps_per_block - 1) / num_warps_per_block;

    conv_transpose2d_warp_kernel<<<num_blocks, threads_per_block>>>(
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
        in_channels_per_group,
        out_channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Warp Shuffle (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}