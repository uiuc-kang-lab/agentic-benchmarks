#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum allowed number of weight elements for constant memory
#define MAX_WEIGHT_SIZE 4096

// Constant memory for the convolution weights
__constant__ float c_weight[MAX_WEIGHT_SIZE];

// Kernel: Utilizes shared memory for input data to reduce global memory access latency
__global__ void shared_memory_conv1d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias,  // Can be nullptr if not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    extern __shared__ float shared_input[];

    int b = blockIdx.x;             // batch index
    int o = blockIdx.y;             // output channel index
    int j = blockIdx.z * blockDim.x + threadIdx.x;  // output spatial position

    if (j >= output_width) return;

    int group_size_out = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = o / group_size_out;
    int o_in_group = o % group_size_out;

    // Load input data into shared memory
    for (int ic = 0; ic < group_in_channels; ++ic) {
        int input_channel = g * group_in_channels + ic;
        int input_index = b * (in_channels * input_width) + input_channel * input_width + j;
        shared_input[threadIdx.x * group_in_channels + ic] = input[input_index];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int i_val = j + padding - k;
        if (i_val % stride != 0) continue;
        int i_idx = i_val / stride;
        if (i_idx < 0 || i_idx >= input_width) continue;
        for (int ic = 0; ic < group_in_channels; ++ic) {
            int weight_index = (ic * group_size_out + o_in_group) * kernel_size + k;
            sum += shared_input[threadIdx.x * group_in_channels + ic] * c_weight[weight_index];
        }
    }

    if (bias != nullptr) {
        sum += bias[o];
    }

    int output_index = b * (out_channels * output_width) + o * output_width + j;
    output[output_index] = sum;
}


torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1);
    int out_channels = group_size_out * groups;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int threads = 256;
    dim3 block(threads);
    dim3 grid(batch_size, out_channels, (output_width + threads - 1) / threads);
    size_t shared_mem_size = threads * in_channels * sizeof(float);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    shared_memory_conv1d_kernel<<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Transposed 1D convolution forward (CUDA)");
}
