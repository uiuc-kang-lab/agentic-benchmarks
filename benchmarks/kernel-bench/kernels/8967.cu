#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Shared memory tile dimensions
#define TILE_DIM 16
#define BLOCK_SIZE 256

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = batch * out_channels * out_h * out_w;
    
    for(int i = tid; i < total; i += stride) {
        int oc = (i / (out_h * out_w)) % out_channels;
        output[i] = bias[oc];
    }
}

__global__ void conv_transposed2d_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_h,
    const int out_w,
    const int in_channels_per_group) {

    extern __shared__ float shared_weight[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int input_idx = bid * blockDim.x + tid;
    
    if(input_idx >= batch * in_channels * in_h * in_w) return;
    
    // Decode input index
    const int iw = input_idx % in_w;
    int tmp = input_idx / in_w;
    const int ih = tmp % in_h;
    tmp = tmp / in_h;
    const int c = tmp % in_channels;
    const int n = tmp / in_channels;
    const int group = c / in_channels_per_group;
    
    // Load input value
    const float x_val = x[input_idx];
    
    // Calculate weight offset for this channel
    const int weight_offset = c * (out_channels_per_group * kernel_h * kernel_w);
    
    // Load weights into shared memory in tiles
    for(int tile = 0; tile < ((kernel_h * kernel_w * out_channels_per_group + TILE_DIM - 1) / TILE_DIM); tile++) {
        const int tile_start = tile * TILE_DIM;
        if(tid < TILE_DIM && tile_start + tid < kernel_h * kernel_w * out_channels_per_group) {
            shared_weight[tid] = weight[weight_offset + tile_start + tid];
        }
        __syncthreads();
        
        const int tile_end = min(TILE_DIM, kernel_h * kernel_w * out_channels_per_group - tile_start);
        
        // Process current tile
        for(int k = 0; k < tile_end; k++) {
            const int weight_idx = tile_start + k;
            const int kh = (weight_idx / out_channels_per_group) / kernel_w;
            const int kw = (weight_idx / out_channels_per_group) % kernel_w;
            const int oc_offset = weight_idx % out_channels_per_group;
            
            const int out_row = ih * stride_h - pad_h + kh * dilation_h;
            const int out_col = iw * stride_w - pad_w + kw * dilation_w;
            
            if(out_row >= 0 && out_row < out_h && out_col >= 0 && out_col < out_w) {
                const int oc = group * out_channels_per_group + oc_offset;
                const int out_idx = n * (groups * out_channels_per_group * out_h * out_w) +
                                  oc * (out_h * out_w) +
                                  out_row * out_w + out_col;
                                  
                atomicAdd(&output[out_idx], x_val * shared_weight[k]);
            }
        }
        __syncthreads();
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with bias
    const int threads_init = 256;
    const int blocks_init = (batch * out_channels * out_h * out_w + threads_init - 1) / threads_init;
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Launch main convolution kernel
    const int total_elements = batch * in_channels * in_h * in_w;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = TILE_DIM * sizeof(float);
    
    conv_transposed2d_shared_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        in_channels / groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Shared Memory (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}