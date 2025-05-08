#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper function for warp-level reduction
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// This kernel computes one output element per block with unrolled loops
__global__ void conv_transposed_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    int index = blockIdx.x;
    int total_output = batch_size * out_channels * out_length / blockDim.x;
    if (index >= total_output) return;

    int out_x = index % out_length;
    int c_out = (index / out_length) % out_channels;
    int n = index / (out_length * out_channels);

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_local = c_out % out_channels_per_group;

    float partial = 0.0f;
    
    // Compute the number of full unrolls (4 iterations per unroll)
    const int UNROLL_FACTOR = 4;
    int total_iters = in_channels_per_group * kernel_size;
    int unrolled_iters = (total_iters / UNROLL_FACTOR) * UNROLL_FACTOR;
    
    // Handle full unrolls
    #pragma unroll
    for (int idx = threadIdx.x; idx < unrolled_iters; idx += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            int current_idx = idx + u * blockDim.x;
            int channel_local = current_idx / kernel_size;
            int k = current_idx % kernel_size;
            int in_channel = group * in_channels_per_group + channel_local;

            int shifted = out_x + padding - k;
            if (shifted % stride == 0) {
                int in_x = shifted / stride;
                if (in_x >= 0 && in_x < in_length) {
                    int input_idx = n * (in_channels * in_length) + in_channel * in_length + in_x;
                    int weight_idx = in_channel * (out_channels_per_group * kernel_size) + 
                                   c_out_local * kernel_size + k;
                    partial += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Handle remaining iterations
    #pragma unroll
    for (int idx = threadIdx.x + unrolled_iters; idx < total_iters; idx += blockDim.x) {
        int channel_local = idx / kernel_size;
        int k = idx % kernel_size;
        int in_channel = group * in_channels_per_group + channel_local;

        int shifted = out_x + padding - k;
        if (shifted % stride == 0) {
            int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                int input_idx = n * (in_channels * in_length) + in_channel * in_length + in_x;
                int weight_idx = in_channel * (out_channels_per_group * kernel_size) + 
                               c_out_local * kernel_size + k;
                partial += input[input_idx] * weight[weight_idx];
            }
        }
    }

    partial = warpReduceSum(partial);

    if (threadIdx.x == 0) {
        float bias_val = 0.0f;
        if (bias != nullptr) {
            bias_val = bias[c_out];
        }
        output[index] = partial + bias_val;
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

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int kernel_size = weight.size(2);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output_tensor = torch::zeros({batch_size, out_channels, out_length}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_ptr = output_tensor.data_ptr<float>();

    int total_output = batch_size * out_channels * out_length / blockDim.x;
    int threads = 32;
    int blocks = total_output;

    auto stream = at::cuda::getCurrentCUDAStream();
    conv_transposed_1d_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
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

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled Transposed 1D convolution forward (CUDA)");
}