#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory and warp-level primitives for reduction
__global__ void depthwise_shared_memory_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_h,
    const int input_w,
    const int out_channels,
    const int output_h,
    const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int channels_per_group
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int oc_batch = blockIdx.z;
    const int b = oc_batch / out_channels;
    const int oc = oc_batch % out_channels;

    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_out >= output_w || h_out >= output_h) return;

    const int in_ch = oc / channels_per_group;
    const int weight_ch = oc % channels_per_group;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        const int h_in = h_out * stride + kh - padding;
        if (h_in >= 0 && h_in < input_h) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in = w_out * stride + kw - padding;
                if (w_in >= 0 && w_in < input_w) {
                    int input_idx = b * (in_channels * input_h * input_w) +
                                    in_ch * (input_h * input_w) +
                                    h_in * input_w + w_in;
                    int weight_idx = in_ch * (channels_per_group * kernel_size * kernel_size) +
                                     weight_ch * (kernel_size * kernel_size) +
                                     kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float block_sum = shared_mem[0];
        if (bias != nullptr) {
            block_sum += bias[oc];
        }
        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      h_out * output_w +
                      w_out;
        output[out_idx] = block_sum;
    }
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
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;

    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 block(TILE_W, TILE_H);
    dim3 grid((output_w + TILE_W - 1) / TILE_W,
              (output_h + TILE_H - 1) / TILE_H,
              batch_size * out_channels);

    size_t shared_mem_size = TILE_W * TILE_H * sizeof(float);

    depthwise_shared_memory_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
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
    m.def("forward", &forward, "Depthwise 2D Convolution with shared memory reduction",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
