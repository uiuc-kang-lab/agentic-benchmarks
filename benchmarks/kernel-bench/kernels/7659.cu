#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// Tunable parameters
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8 
#define BLOCK_SIZE_Z 4
#define WARPS_PER_BLOCK 8
#define SHARED_MEM_SIZE 4096

__global__ void optimized_conv3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int batch_size, const int in_channels, const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int out_depth, const int out_height, const int out_width,
    const int stride, const int padding, const int dilation, const int groups) {

    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    float* weight_tile = shared_mem + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int w_out = blockIdx.x * blockDim.x + tx;
    const int h_out = blockIdx.y * blockDim.y + ty;
    const int d_out = blockIdx.z * blockDim.z + tz;

    if (w_out >= out_width || h_out >= out_height || d_out >= out_depth) 
        return;

    const int warp_id = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 32;
    const int lane_id = threadIdx.x % 32;

    #pragma unroll 4
    for (int b = 0; b < batch_size; ++b) {
        for (int c_out_block = 0; c_out_block < out_channels; c_out_block += WARPS_PER_BLOCK) {
            const int c_out = c_out_block + warp_id;
            if (c_out >= out_channels) continue;

            float sum = 0.0f;
            const int group = c_out / (out_channels / groups);
            const int in_channels_per_group = in_channels / groups;

            const int d_in_start = d_out * stride - padding;
            const int h_in_start = h_out * stride - padding;
            const int w_in_start = w_out * stride - padding;

            if (bias != nullptr && lane_id == 0) {
                sum += bias[c_out];
            }

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                const int in_c = group * in_channels_per_group + ic;

                __syncthreads();
                for (int i = warp_id; i < kernel_d * kernel_h * kernel_w; i += WARPS_PER_BLOCK) {
                    const int kd = i / (kernel_h * kernel_w);
                    const int kh = (i / kernel_w) % kernel_h;
                    const int kw = i % kernel_w;
                    
                    const int d_in = d_in_start + kd * dilation;
                    const int h_in = h_in_start + kh * dilation;
                    const int w_in = w_in_start + kw * dilation;

                    if (d_in >= 0 && d_in < in_depth && 
                        h_in >= 0 && h_in < in_height && 
                        w_in >= 0 && w_in < in_width) {
                        input_tile[i] = input[((b * in_channels + in_c) * in_depth + d_in) * 
                                             in_height * in_width + h_in * in_width + w_in];
                        weight_tile[i] = weight[(((c_out * in_channels_per_group) + ic) * kernel_d + kd) *
                                              kernel_h * kernel_w + kh * kernel_w + kw];
                    } else {
                        input_tile[i] = 0.0f;
                        weight_tile[i] = 0.0f;
                    }
                }
                __syncthreads();

                #pragma unroll
                for (int i = 0; i < kernel_d * kernel_h * kernel_w; ++i) {
                    sum += input_tile[i] * weight_tile[i];
                }
            }

            if (w_out < out_width && h_out < out_height && d_out < out_depth) {
                const int output_idx = ((b * out_channels + c_out) * out_depth + d_out) * 
                                     out_height * out_width + h_out * out_width + w_out;
                output[output_idx] = sum;
            }
        }
    }
}

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
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "Bias must be a CUDA tensor");

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);
    
    const auto out_channels = weight.size(0);
    const auto kernel_d = weight.size(2);
    const auto kernel_h = weight.size(3);
    const auto kernel_w = weight.size(4);
    
    const auto out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const auto out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        (out_depth + block.z - 1) / block.z
    );

    const int shared_mem_size = (kernel_d * kernel_h * kernel_w) * sizeof(float) * 2;

    optimized_conv3d_kernel<<<grid, block, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation, groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D convolution forward (CUDA)");
}