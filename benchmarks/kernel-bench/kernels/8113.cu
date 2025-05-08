#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that operates on each output element individually
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
    int total_output = batch_size * out_channels * out_length;
    if (index >= total_output) return;

    int out_x = index % out_length;
    int c_out = (index / out_length) % out_channels;
    int n = index / (out_length * out_channels);

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_local = c_out % out_channels_per_group;

    float partial = 0.0f;
    for (int i = threadIdx.x; i < in_channels_per_group * kernel_size; i += blockDim.x) {
        int channel_local = i / kernel_size;
        int k = i % kernel_size;
        int in_channel = group * in_channels_per_group + channel_local;

        int shifted = out_x + padding - k;
        if (shifted % stride == 0) {
            int in_x = shifted / stride;
            if (in_x >= 0 && in_x < in_length) {
                int input_idx = n * in_channels * in_length + in_channel * in_length + in_x;
                int weight_idx = in_channel * out_channels_per_group * kernel_size + c_out_local * kernel_size + k;
                partial += input[input_idx] * weight[weight_idx];
            }
        }
    }

    __shared__ float shared_data[32];
    shared_data[threadIdx.x] = partial;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x; ++i)
            partial += shared_data[i];

        // Add bias
        if (bias) partial += bias[c_out];

        output[index] = partial;
    }
}

// Host function with streams to overlap memory transfer and computation
// This includes asynchronous memory operations and kernel launches
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
    int threads = 32; //Number of threads (one warp)
    int blocks = total_output; //Number of blocks

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously transfer data
    cudaMemcpyAsync(input_ptr, input.data_ptr<float>(), sizeof(float) * input.numel(), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(weight_ptr, weight.data_ptr<float>(), sizeof(float) * weight.numel(), cudaMemcpyHostToDevice, stream2);
    if (bias.has_value()) {
        cudaMemcpyAsync(bias_ptr, bias.value().data_ptr<float>(), sizeof(float) * bias.value().numel(), cudaMemcpyHostToDevice, stream2);
    }

    // Launch kernel
    conv_transposed_1d_kernel<<<blocks, threads, 0, stream1>>>(
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

    cudaMemcpyAsync(output_tensor.data_ptr<float>(), output_ptr, sizeof(float) * output_tensor.numel(), cudaMemcpyDeviceToHost, stream1);

    // Synchronize
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Overlap Memory Transfer and Compute (CUDA)");
}