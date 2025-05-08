#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Threshold for using optimized kernel vs ATen implementation
#define OPTIMIZED_KERNEL_THRESHOLD 64

// CUDA kernel with shared memory and warp-level optimizations
__global__ void optimized_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int iD, int iH, int iW,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups) {

    extern __shared__ float shared_mem[];
    
    // Only using shared memory for weight tiles
    float* shared_weight = shared_mem;

    int total_elements = batch * out_channels * outD * outH * outW;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Decode output index
    int w = tid % outW;
    int tmp = tid / outW;
    int h = tmp % outH;
    tmp = tmp / outH;
    int d = tmp % outD;
    tmp = tmp / outD;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Calculate group index and local channel index
    int group_idx = oc / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    float sum = 0.0f;
    
    if (groups == 1) {
        // Use warp-level parallelism for channels
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        for (int ic = warp_id; ic < in_channels; ic += blockDim.x/32) {
            // Load input/weight tiles to shared memory
            if (lane_id < kD*kH*kW) {
                int kid = lane_id / (kH*kW);
                int khw = lane_id % (kH*kW);
                int kh = khw / kW;
                int kw = khw % kW;
                
                shared_weight[lane_id] = weight[((ic * out_channels + oc) * kD + kid) * kH*kW + khw];
            }
            __syncwarp();

            #pragma unroll 4
            for (int kd = 0; kd < kD; ++kd) {
                int id = d + pad_d - kd;
                if ((id % stride_d) == 0) {
                    id /= stride_d;
                    if (id >= 0 && id < iD) {
                        #pragma unroll 4
                        for (int kh = 0; kh < kH; ++kh) {
                            int ih = h + pad_h - kh;
                            if ((ih % stride_h) == 0) {
                                ih /= stride_h;
                                if (ih >= 0 && ih < iH) {
                                    #pragma unroll 4
                                    for (int kw = 0; kw < kW; ++kw) {
                                        int iw = w + pad_w - kw;
                                        if ((iw % stride_w) == 0) {
                                            iw /= stride_w;
                                            if (iw >= 0 && iw < iW) {
                                                int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
                                                int weight_idx = ((kd * kH) + kh) * kW + kw;
                                                sum += input[input_idx] * shared_weight[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncwarp();
        }
    }

    // Warp-level reduction for final sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[(((b * out_channels + oc) * outD + d) * outH + h) * outW + w] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    if (x.size(1) <= OPTIMIZED_KERNEL_THRESHOLD) {
        std::vector<int64_t> dilation = {1, 1, 1};
        return at::conv_transpose3d(x, weight, 
            bias ? *bias : torch::Tensor(),
            stride, padding, output_padding, 
            groups, dilation);
    }
    
    auto output = torch::zeros({x.size(0), weight.size(1) * groups,
                               (x.size(2) - 1) * stride[0] - 2 * padding[0] + weight.size(2) + output_padding[0],
                               (x.size(3) - 1) * stride[1] - 2 * padding[1] + weight.size(3) + output_padding[1],
                               (x.size(4) - 1) * stride[2] - 2 * padding[2] + weight.size(4) + output_padding[2]},
                               x.options());
    
    int total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    int shared_mem_size = threads * sizeof(float) * 2;
    
    optimized_conv_transpose3d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        x.size(0), x.size(1), output.size(1),
        x.size(2), x.size(3), x.size(4),
        weight.size(2), weight.size(3), weight.size(4),
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output.size(2), output.size(3), output.size(4),
        groups);
    
    return output;
}