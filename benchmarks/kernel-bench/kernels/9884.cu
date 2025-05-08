#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[kernel_size * kernel_size];

    int total_elements = batch_size * out_channels * output_h * output_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int w_out = idx % output_w;
    idx /= output_w;
    int h_out = idx % output_h;
    idx /= output_h;
    int oc = idx % out_channels;
    int b = idx / out_channels;

    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Load weights into shared memory - each thread helps load the kernel
    for (int i = threadIdx.x; i < kernel_size * kernel_size; i += blockDim.x) {
        int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size)
                        + weight_ch * (kernel_size * kernel_size)
                        + i;
        shared_weight[i] = weight[weight_idx];
    }

    // Calculate input region for this thread
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    // Load input region into shared memory
    for (int i = threadIdx.x; i < (kernel_size * kernel_size); i += blockDim.x) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int h_in = h_in_start + kh;
        int w_in = w_in_start + kw;

        float val = 0.0f;
        if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
            int input_idx = b * (in_channels * input_h * input_w)
                           + in_ch * (input_h * input_w)
                           + h_in * input_w
                           + w_in;
            val = input[input_idx];
        }
        shared_input[i] = val;
    }

    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        sum += shared_input[i] * shared_weight[i];
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[b * out_channels * output_h * output_w +
           oc * output_h * output_w +
           h_out * output_w +
           w_out] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;

    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == out_channels, "Bias size mismatch");
    }

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int threads = 256;
    int blocks = (batch_size * out_channels * output_h * output_w + threads - 1) / threads;
    
    // Shared memory size calculation
    int shared_mem_size = (threads + kernel_size - 1) * (threads + kernel_size - 1) * sizeof(float) // for input
                         + kernel_size * kernel_size * sizeof(float); // for weights

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_h,
        input_w,
        out_channels,
        output_h,
        output_w,
        kernel_size,
        stride,
        padding,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}