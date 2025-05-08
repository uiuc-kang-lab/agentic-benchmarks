#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__ __device__ void compute_output_indices(
    int idx,
    int output_w,
    int output_h,
    int out_channels,
    int& b,
    int& oc,
    int& h_out,
    int& w_out
) {
    w_out = idx % output_w;
    idx /= output_w;
    h_out = idx % output_h;
    idx /= output_h;
    oc = idx % out_channels;
    b = idx / out_channels;
}

__forceinline__ __device__ float load_input(
    const float* __restrict__ input,
    int b,
    int in_ch,
    int h_in,
    int w_in,
    int input_h,
    int input_w,
    int in_channels
) {
    if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
        const int idx = b * in_channels * input_h * input_w
                      + in_ch * input_h * input_w
                      + h_in * input_w
                      + w_in;
        return __ldg(&input[idx]);
    }
    return 0.0f;
}

__forceinline__ __device__ float load_weight(
    const float* __restrict__ weight,
    int in_ch,
    int weight_ch,
    int kh,
    int kw,
    int kernel_size,
    int channels_per_group
) {
    const int idx = in_ch * channels_per_group * kernel_size * kernel_size
                  + weight_ch * kernel_size * kernel_size
                  + kh * kernel_size
                  + kw;
    return __ldg(&weight[idx]);
}

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    constexpr int unroll_factor = 4;
    constexpr int TILE_SIZE = 16;
    const int total_elements = batch_size * out_channels * output_h * output_w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    __shared__ float weight_shared[TILE_SIZE][TILE_SIZE];
    
    int b, oc, h_out, w_out;
    compute_output_indices(
        idx,
        output_w,
        output_h,
        out_channels,
        b,
        oc,
        h_out,
        w_out
    );

    const int in_ch = oc / channels_per_group;
    const int weight_ch = oc % channels_per_group;
    float sum = 0.0f;
    
    // Cooperatively load weights into shared memory
    const int tid = threadIdx.x;
    if (tid < kernel_size * kernel_size) {
        const int kh = tid / kernel_size;
        const int kw = tid % kernel_size;
        weight_shared[kh][kw] = load_weight(weight, in_ch, weight_ch, kh, kw,
                                          kernel_size, channels_per_group);
    }
    __syncthreads();

    #pragma unroll unroll_factor
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int h_in = h_out * stride + kh - padding;
        #pragma unroll unroll_factor
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int kernel_offset = kh * kernel_size + kw;
            const int w_in = w_out * stride + kw - padding;

            sum += load_input(input, b, in_ch, h_in, w_in,
                             input_h, input_w, in_channels)
                 * load_weight(weight, in_ch, weight_ch, kh, kw,
                             kernel_size, channels_per_group);
        }
    }

    if (bias) {
        sum += __ldg(&bias[oc]);
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

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;

    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w},
                              input.options());

    const int total_elements = output.numel();
    const int threads = 512;
    const int blocks = (total_elements + threads - 1) / threads;

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
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
    m.def("forward", &forward, "Depthwise 2D Convolution (Optimized)",
          py::arg("input"), py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"));
}