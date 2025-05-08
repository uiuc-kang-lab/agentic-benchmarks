#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using shared memory for reduction
__global__ void conv_transposed_1d_shared_kernel(
    const float* __restrict__ input,   // [N, in_channels, input_width]
    const float* __restrict__ weight,  // [in_channels, out_channels, kernel_size]
    const float* __restrict__ bias,    // [out_channels] or nullptr
    float* __restrict__ output,        // [N, out_channels, output_width]
    int N,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int ox = blockIdx.x * blockDim.x + tid;
    if (ox >= output_width) return;
    int oc = blockIdx.y;
    int n = blockIdx.z;

    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;
    int ic_start = group * in_channels_per_group;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int global_ic = ic_start + ic;
        for (int k = 0; k < kernel_size; k++) {
            int temp = ox + padding - k;
            if (temp < 0) continue;
            if ((temp % stride) != 0) continue;
            int ix = temp / stride;
            if (ix < 0 || ix >= input_width) continue;

            int input_index = n * (in_channels * input_width) + global_ic * input_width + ix;
            int weight_index = global_ic * (out_channels * kernel_size) + oc * kernel_size + k;
            sum += input[input_index] * weight[weight_index];
        }
    }

    shared_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[n * (out_channels * output_width) + oc * output_width + ox] = shared_data[0];
    }
}

// Host function interfaced via pybind11 that prepares inputs and launches the kernel
torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    auto input_sizes = input.sizes();
    int N = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_width = input_sizes[2];

    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];

    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({N, out_channels, output_width}, input.options());

    const int threads = 256;
    const int blocks_x = (output_width + threads - 1) / threads;
    dim3 blocks(blocks_x, out_channels, N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t shared_mem_size = threads * sizeof(float);
    conv_transposed_1d_shared_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Transposed 1D convolution forward with shared memory (CUDA)");
}