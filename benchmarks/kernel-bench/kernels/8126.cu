#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_SIZE = 128;
const int TILE_SIZE = 16;

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
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
    int groups) {
    
    extern __shared__ float shared_buffer[];
    float* input_tile = shared_buffer;
    float* weight_tile = &shared_buffer[TILE_SIZE * BLOCK_SIZE];

    int tile_idx = threadIdx.x / TILE_SIZE;
    int lane = threadIdx.x % TILE_SIZE;
    
    int output_idx = blockIdx.x * TILE_SIZE + lane;
    if (output_idx >= batch_size * out_channels * out_length) return;

    int n = output_idx / (out_channels * out_length);
    int c_out = (output_idx / out_length) % out_channels;
    int out_x = output_idx % out_length;

    int group = c_out / (out_channels / groups);
    int in_chan_per_group = in_channels / groups;
    int out_chan_per_group = out_channels / groups;
    
    float acc = 0.0f;
    
    for (int tile = 0; tile < (kernel_size * in_chan_per_group + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int load_idx = tile * TILE_SIZE + threadIdx.x;
        if (load_idx < kernel_size * in_chan_per_group) {
            int k = load_idx % kernel_size;
            int in_c = group * in_chan_per_group + (load_idx / kernel_size);
            
            int shifted = out_x + padding - k;
            if (shifted % stride == 0) {
                int in_x = shifted / stride;
                if (in_x >= 0 && in_x < in_length)
                    input_tile[load_idx % (TILE_SIZE * BLOCK_SIZE)] = input[n * (in_channels * in_length) + in_c * in_length + in_x];
                else
                    input_tile[load_idx % (TILE_SIZE * BLOCK_SIZE)] = 0.0f;
            } else {
                input_tile[load_idx % (TILE_SIZE * BLOCK_SIZE)] = 0.0f;
            }
            weight_tile[load_idx % (TILE_SIZE * BLOCK_SIZE)] = weight[in_c * (out_chan_per_group * kernel_size) + (c_out % out_chan_per_group) * kernel_size + k];
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += input_tile[tile * TILE_SIZE + i] * weight_tile[tile * TILE_SIZE + i];
        }
        __syncthreads();
    }

    acc = warpReduceSum(acc);
    
    if (threadIdx.x % warpSize == 0) {
        if (bias) acc += bias[c_out];
        output[output_idx] = acc;
    }
}

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
    if (bias.has_value()) CHECK_INPUT(bias.value());

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int kernel_size = weight.size(2);
    
    int out_channels = weight.size(1) * groups;
    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_length}, input.options());
    
    int num_outputs = batch_size * out_channels * out_length;
    int shared_mem = (TILE_SIZE * BLOCK_SIZE + TILE_SIZE * BLOCK_SIZE) * sizeof(float);
    
    conv_transposed_1d_kernel<<<
        (num_outputs + TILE_SIZE - 1) / TILE_SIZE,
        BLOCK_SIZE,
        shared_mem,
        at::cuda::getCurrentCUDAStream()
    >>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        groups
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Optimized Transposed 1D Conv");
}