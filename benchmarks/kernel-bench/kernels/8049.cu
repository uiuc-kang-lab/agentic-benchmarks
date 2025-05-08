#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + BLOCK_SIZE;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Load input and weight data into shared memory
    if (tid < input_length) {
        shared_input[tid] = input[tid];
    }
    if (tid < kernel_size) {
        shared_weight[tid] = weight[tid];
    }
    __syncthreads();

    // Calculate output position
    const int out_idx = bid * BLOCK_SIZE + tid;
    if (out_idx >= output_length) return;

    float result = 0.0f;
    
    // Compute convolution with shared memory
    #pragma unroll
    for (int k = 0; k < kernel_size; k++) {
        const int in_pos = (out_idx + padding - k) / stride;
        if (in_pos >= 0 && in_pos < input_length) {
            result += shared_input[in_pos] * shared_weight[kernel_size - 1 - k];
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        result += __shfl_down_sync(0xffffffff, result, offset);
    }

    // First thread in warp writes result
    if (tid % WARP_SIZE == 0) {
        if (bias != nullptr) {
            result += bias[out_idx % out_channels];
        }
        output[out_idx] = result;
    }
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

    auto input_size = x.sizes();
    auto weight_size = weight.sizes();
    
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int input_length = input_size[2];
    int out_channels = weight_size[0];
    int kernel_size = weight_size[2];
    
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_length},
                              x.options());

    const int shared_mem_size = (BLOCK_SIZE + kernel_size) * sizeof(float);
    const dim3 blocks((output_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 threads(BLOCK_SIZE);

    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose1d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed 1D convolution forward (CUDA)");
}