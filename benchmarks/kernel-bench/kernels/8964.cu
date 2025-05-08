#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch, int out_channels, int out_h, int out_w) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (index < total) {
        int oc = (index / (out_h * out_w)) % out_channels;
        output[index] = bias[oc];
    }
}

__global__ void conv_transposed2d_shared_reduce_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int in_channels, int in_h, int in_w,
    int out_channels_per_group, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups,
    int out_h, int out_w, int in_channels_per_group) {

    extern __shared__ float shared_mem[];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * in_channels * in_h * in_w) return;
    
    // Decode input index
    int iw = tid % in_w;
    int tmp = tid / in_w;
    int ih = tmp % in_h;
    tmp = tmp / in_h;
    int c_in = tmp % in_channels;
    int n = tmp / in_channels;

    int group = c_in / in_channels_per_group;
    float x_val = __ldg(&x[tid]);

    // Warp-based reduction variables
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);
    const int lane_id = warp.thread_rank();

    for (int kh = 0; kh < kernel_h; ++kh) {
        int out_row = ih * stride_h - pad_h + kh * dilation_h;
        if (out_row < 0 || out_row >= out_h) continue;
        
        for (int kw = 0; kw < kernel_w; ++kw) {
            int out_col = iw * stride_w - pad_w + kw * dilation_w;
            if (out_col < 0 || out_col >= out_w) continue;

            for (int oc_offset = 0; oc_offset < out_channels_per_group; ++oc_offset) {
                // Calculate contribution
                int oc = group * out_channels_per_group + oc_offset;
                int weight_idx = c_in * (out_channels_per_group * kernel_h * kernel_w)
                               + oc_offset * (kernel_h * kernel_w)
                               + kh * kernel_w + kw;
                float contrib = x_val * __ldg(&weight[weight_idx]);

                // Calculate output index
                int out_idx = n * out_channels * out_h * out_w
                             + oc * out_h * out_w
                             + out_row * out_w + out_col;

                // Warp-level reduction
                float sum = cg::reduce(warp, contrib, cg::plus<float>());
                
                // First thread in warp performs atomic add
                if (lane_id == 0) {
                    atomicAdd(&output[out_idx], sum);
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x, at::Tensor weight, c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, int groups) {

    x = x.contiguous();
    weight = weight.contiguous();
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    // Dimension setup
    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2), in_w = x.size(3);
    int kernel_h = weight.size(2), kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int in_channels_per_group = in_channels / groups;

    // Calculate output dimensions
    int stride_h = stride[0], stride_w = stride[1];
    int pad_h = padding[0], pad_w = padding[1];
    int dilation_h = dilation[0], dilation_w = dilation[1];
    int out_h = (in_h-1)*stride_h - 2*pad_h + dilation_h*(kernel_h-1) + 1;
    int out_w = (in_w-1)*stride_w - 2*pad_w + dilation_w*(kernel_w-1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with bias values
    int total_output = batch * out_channels * out_h * out_w;
    dim3 block_init(512);
    dim3 grid_init((total_output + block_init.x - 1) / block_init.x);
    initialize_output_kernel<<<grid_init, block_init>>>(
        output.data_ptr<float>(), bias.value().data_ptr<float>(),
        batch, out_channels, out_h, out_w);

    // Launch main convolution kernel
    int total_input = batch * in_channels * in_h * in_w;
    dim3 block_conv(256); // Favor more threads for better occupancy
    dim3 grid_conv((total_input + block_conv.x - 1) / block_conv.x);

    // 32 threads per warp * sizeof(float) for temp storage
    size_t shared_size = 0; // No explicit shared memory needed for warp reduce
    
    conv_transposed2d_shared_reduce_kernel<<<grid_conv, block_conv, shared_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, in_h, in_w,
        out_channels_per_group, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
        out_h, out_w, in_channels_per_group);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Conv with Warp Reduction");
}
