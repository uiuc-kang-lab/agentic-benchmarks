#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized warp-level reduction using shuffle instructions
__forceinline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized warp-level data sharing
__forceinline__ __device__ float warpBroadcast(float val, int srcLane) {
    return __shfl_sync(0xffffffff, val, srcLane);
}

__global__ void conv_transposed_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_length,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warpSize = 32;
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;

    // Calculate output position
    const int out_idx = bid;
    if (out_idx >= batch_size * out_channels * out_length) return;

    const int out_x = out_idx % out_length;
    const int c_out = (out_idx / out_length) % out_channels;
    const int n = out_idx / (out_length * out_channels);
    
    // Calculate group information
    const int group = c_out / out_channels_per_group;
    const int c_out_local = c_out % out_channels_per_group;

    // Compute partial sum using warp-level parallelism and shared memory tiling
    float partial_sum = 0.0f;
    
    // Load weights into shared memory
    const int weight_offset = (group * in_channels_per_group) * (out_channels_per_group * kernel_size) + 
                            c_out_local * kernel_size;
    if (tid < kernel_size) {
        shared_weight[tid] = weight[weight_offset + tid];
    }
    
    #pragma unroll
    for (int k = tid; k < kernel_size; k += warpSize) {
        const int shifted = out_x + padding - k;
        if (shifted % stride == 0) {
            const int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                const int in_ch_start = group * in_channels_per_group;
                const int in_ch_end = (group + 1) * in_channels_per_group;
                
                #pragma unroll
                for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
                    const int input_idx = n * (in_channels * in_length) + in_ch * in_length + in_x;
                    const int weight_idx = (in_ch - in_ch_start) * (out_channels_per_group * kernel_size) + 
                                         c_out_local * kernel_size + k;
                    
                    partial_sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Perform warp-level reduction
    partial_sum = warpReduceSum(partial_sum);

    // First thread in warp writes result
    if (tid == 0) {
        if (bias != nullptr) {
            partial_sum += bias[c_out];
        }
        output[out_idx] = partial_sum;
    }
}

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_length = input.size(2);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;
    const int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());

    const dim3 threads(32); // One warp per block
    const dim3 blocks(batch_size * out_channels * out_length);

    auto stream = at::cuda::getCurrentCUDAStream();
    conv_transposed_1d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Optimized Transposed 1D convolution forward (CUDA)");
}