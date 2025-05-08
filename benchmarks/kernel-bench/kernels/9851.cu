#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for the output tile
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

// Kernel that combines shared memory tiling with warp-level reduction for efficiency
__global__ void optimized_depthwise_conv2d_kernel(
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
    const int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / warpSize;
    int lane = globalThreadId % warpSize;

    if (warpId >= batch_size * out_channels * output_h * output_w) return;

    // Decode warpId into output indices: (b, oc, out_h, out_w)
    int tmp = warpId;
    int out_w_idx = tmp % output_w;
    tmp /= output_w;
    int out_h_idx = tmp % output_h;
    tmp /= output_h;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    int in_ch = oc / channels_per_group;
    
    // Compute starting input coordinates for this output pixel
    int in_y_origin = out_h_idx * stride - padding;
    int in_x_origin = out_w_idx * stride - padding;

    // Allocate shared memory
    extern __shared__ float shared_mem[];
    float* s_weight = shared_mem;  // Size: kernel_size * kernel_size

    // Load the weight kernel into shared memory
    int total_weight = kernel_size * kernel_size;
    for (int i = lane; i < total_weight; i += warpSize) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            (oc % channels_per_group) * (kernel_size * kernel_size) + i
        ];
    }

    __syncthreads();

    int kernel_area = kernel_size * kernel_size;
    float sum = 0.0f;

    // Each thread in the warp covers a subset of the kernel elements
    for (int i = lane; i < kernel_area; i += warpSize) {
        int ky = i / kernel_size;
        int kx = i % kernel_size;
        int in_y = in_y_origin + ky;
        int in_x = in_x_origin + kx;
        float input_val = 0.0f;
        if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            in_y * input_w + in_x;
            input_val = input[input_idx];
        }
        float wt = s_weight[i];
        sum += input_val * wt;
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the result
    if (lane == 0) {
        if(bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_h_idx * output_w + out_w_idx;
        output[out_idx] = sum;
    }
}

// Forward function callable from Python
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
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

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

    int total_output = batch_size * out_channels * output_h * output_w;
    const int warpSize = 32;
    int total_threads = total_output * warpSize;  // one warp per output element
    int block_size = 128;  // e.g., 4 warps per block
    int grid_size = (total_threads + block_size - 1) / block_size;

    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    size_t shared_mem_bytes = kernel_size * kernel_size * sizeof(float);

    optimized_depthwise_conv2d_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
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
    m.def("forward", &forward, "Optimized Depthwise 2D Convolution (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}