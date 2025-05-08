#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_shared_warp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int groups) {

    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    const int total_outputs = batch_size * out_channels * out_height * out_width;
    const int outputs_per_block = (total_outputs + gridDim.x - 1) / gridDim.x;
    const int start_output = blockIdx.x * outputs_per_block;
    const int end_output = min(start_output + outputs_per_block, total_outputs);

    float* warp_sums = shared_mem;

    for (int output_idx = start_output + warp_id; output_idx < end_output; output_idx += warps_per_block) {
        const int w = output_idx % out_width;
        const int h = (output_idx / out_width) % out_height;
        const int oc = (output_idx / (out_width * out_height)) % out_channels;
        const int b = output_idx / (out_channels * out_height * out_width);

        const int group = oc / (out_channels / groups);
        const int in_channels_per_group = in_channels / groups;

        float partial_sum = 0.0f;

        for (int ic = lane_id; ic < in_channels_per_group; ic += WARP_SIZE) {
            const int input_channel = group * in_channels_per_group + ic;
            
            for (int kh = 0; kh < kernel_height; kh++) {
                const int in_h = h * stride - padding + kh;
                if (in_h >= 0 && in_h < in_height) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        const int in_w = w * stride - padding + kw;
                        if (in_w >= 0 && in_w < in_width) {
                            const int in_idx = ((b * in_channels + input_channel) * in_height + in_h) * in_width + in_w;
                            const int weight_idx = ((oc * in_channels_per_group + ic) * kernel_height + kh) * kernel_width + kw;
                            partial_sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (lane_id == 0) {
            warp_sums[warp_id] = partial_sum;
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < warps_per_block) {
            float sum = warp_sums[lane_id];
            
            #pragma unroll
            for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (lane_id == 0) {
                if (bias != nullptr) {
                    sum += bias[oc];
                }
                output[output_idx] = sum;
            }
        }
        __syncthreads();
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    const int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    const int num_blocks = 112;

    conv2d_shared_warp_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Shared Memory and Warp Reduction");
}