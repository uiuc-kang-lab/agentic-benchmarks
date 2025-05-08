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
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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
    // Using shared memory for intermediate results
    __shared__ float shared_reduction[4]; // One per warp in the block
    
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    
    int index = blockIdx.x;
    int total_output = batch_size * out_channels * out_length;
    if (index >= total_output) return;

    int out_x = index % out_length;
    int c_out = (index / out_length) % out_channels;
    int n = index / (out_length * out_channels);

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_local = c_out % out_channels_per_group;

    float partial_sum = 0.0f;
    int total_iters = in_channels_per_group * kernel_size;
    
    // Distribute work across all threads in the block
    for (int idx = tid; idx < total_iters; idx += blockDim.x) {
        int channel_local = idx / kernel_size;
        int k = idx % kernel_size;
        int in_channel = group * in_channels_per_group + channel_local;

        int shifted = out_x + padding - k;
        if (shifted % stride == 0) {
            int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                int input_idx = n * (in_channels * in_length) + in_channel * in_length + in_x;
                int weight_idx = in_channel * (out_channels_per_group * kernel_size) + c_out_local * kernel_size + k;
                partial_sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Perform warp-level reduction
    partial_sum = warpReduceSum(partial_sum);

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_reduction[warp_id] = partial_sum;
    }
    
    __syncthreads();

    // First warp reduces results from all warps
    if (warp_id == 0) {
        float final_sum = 0.0f;
        if (lane_id < 4) { // Number of warps in block
            final_sum = shared_reduction[lane_id];
        }
        final_sum = warpReduceSum(final_sum);

        if (lane_id == 0) {
            if (bias != nullptr) {
                final_sum += bias[c_out];
            }
            output[index] = final_sum;
        }
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

    int total_output = batch_size * out_channels * out_length;
    int threads = 128; // Increased block size to 128 threads (4 warps)
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
    m.def("forward", &forward, "Block size optimized Transposed 1D convolution forward (CUDA)");
}