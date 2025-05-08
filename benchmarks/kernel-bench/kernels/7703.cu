#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_D 4
#define TILE_H 4
#define TILE_W 4
#define CHANNELS_PER_THREAD 2

__global__ void conv3d_tiled_stride_kernel(
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
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation
) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tidz = threadIdx.z;
    
    const int block_out_d = blockIdx.x * TILE_D;
    const int block_out_h = blockIdx.y * TILE_H;
    const int block_out_w = blockIdx.z * TILE_W;

    const int out_channel_group = blockIdx.w % ((out_channels + CHANNELS_PER_THREAD - 1) / CHANNELS_PER_THREAD);
    const int batch_group = blockIdx.w / ((out_channels + CHANNELS_PER_THREAD - 1) / CHANNELS_PER_THREAD);

    float results[CHANNELS_PER_THREAD] = {0};

    for (int ic = 0; ic < in_channels; ++ic) {
        #pragma unroll
        for (int kd = 0; kd < kernel_d; ++kd) {
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    float filter_vals[CHANNELS_PER_THREAD];
                    
                    #pragma unroll
                    for (int oc = 0; oc < CHANNELS_PER_THREAD; ++oc) {
                        const int weight_idx = (
                            ((out_channel_group * CHANNELS_PER_THREAD + oc) * in_channels + ic) * kernel_d + kd
                        ) * kernel_h * kernel_w + kh * kernel_w + kw;
                        filter_vals[oc] = weight[weight_idx];
                    }

                    #pragma unroll
                    for (int dd = 0; dd < TILE_D; ++dd) {
                        const int od = block_out_d + dd;
                        if (od >= out_depth) continue;
                        
                        const int id = od * stride - padding + kd * dilation;
                        if (id < 0 || id >= in_depth) continue;

                        #pragma unroll
                        for (int dh = 0; dh < TILE_H; ++dh) {
                            const int oh = block_out_h + dh;
                            if (oh >= out_height) continue;
                            
                            const int ih = oh * stride - padding + kh * dilation;
                            if (ih < 0 || ih >= in_height) continue;

                            #pragma unroll
                            for (int dw = 0; dw < TILE_W; ++dw) {
                                const int ow = block_out_w + dw;
                                if (ow >= out_width) continue;
                                
                                const int iw = ow * stride - padding + kw * dilation;
                                if (iw < 0 || iw >= in_width) continue;

                                const float input_val = input[
                                    ((batch_group * in_channels + ic) * in_depth + id) * 
                                    in_height * in_width + ih * in_width + iw
                                ];

                                #pragma unroll
                                for (int oc = 0; oc < CHANNELS_PER_THREAD; ++oc) {
                                    results[oc] += input_val * filter_vals[oc];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int oc = 0; oc < CHANNELS_PER_THREAD; ++oc) {
        const int out_c = out_channel_group * CHANNELS_PER_THREAD + oc;
        if (out_c >= out_channels) continue;

        #pragma unroll
        for (int dd = 0; dd < TILE_D; ++dd) {
            const int od = block_out_d + dd;
            if (od >= out_depth) continue;

            #pragma unroll
            for (int dh = 0; dh < TILE_H; ++dh) {
                const int oh = block_out_h + dh;
                if (oh >= out_height) continue;

                #pragma unroll
                for (int dw = 0; dw < TILE_W; ++dw) {
                    const int ow = block_out_w + dw;
                    if (ow >= out_width) continue;

                    const int output_idx = (
                        ((batch_group * out_channels + out_c) * out_depth + od) * 
                        out_height + oh
                    ) * out_width + ow;
                    
                    output[output_idx] = results[oc] + (bias ? bias[out_c] : 0.0f);
                }
            }
        }
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "Only groups=1 supported");
    auto bias = bias_opt.value_or(at::Tensor());

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(TILE_W, TILE_H, TILE_D);
    dim3 blocks(
        (out_depth + TILE_D - 1) / TILE_D,
        (out_height + TILE_H - 1) / TILE_H,
        (out_width + TILE_W - 1) / TILE_W
    );

    const int channel_groups = (out_channels + CHANNELS_PER_THREAD - 1) / CHANNELS_PER_THREAD;
    const int total_blocks_w = channel_groups * batch_size;

    conv3d_tiled_stride_kernel<<<dim4(blocks.x, blocks.y, blocks.z, total_blocks_w), threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled 3D convolution with strided processing (CUDA)");
}
