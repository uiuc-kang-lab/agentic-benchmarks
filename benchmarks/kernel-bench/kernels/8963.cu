#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

// Initialize output tensor using 3D grid indexing.
// Grid dims: (ceil(out_w/tx), ceil(out_h/ty), batch * out_channels).
__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z; // Combined index for batch and channel
    int n = bc / out_channels;
    int oc = bc % out_channels;

    if (col < out_w && row < out_h) {
        int index = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + row * out_w + col;
        output[index] = bias[oc];
    }
}

// Scatter-based 2D transposed convolution kernel using atomicAdd with 3D grid indexing.
// Grid dims: (ceil(in_w/tx), ceil(in_h/ty), batch * in_channels).
__global__ void conv_transposed2d_scatter_atomic_kernel(
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
    const int out_w) {
    
    // Shared memory for input and weight tiles
    __shared__ float s_x[16][16];
    __shared__ float s_weight[16][16];
    
    // Decode combined batch-channel index from gridDim.z
    int bc = blockIdx.z;
    int n = bc / in_channels;
    int c = bc % in_channels;

    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (iw < in_w && ih < in_h) {
        int input_index = n * (in_channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
        s_x[threadIdx.y][threadIdx.x] = x[input_index];
    } else {
        s_x[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    int in_channels_per_group = in_channels / groups;
    int group = c / in_channels_per_group;
    
    __syncthreads();
    
    // Iterate over kernel spatial dimensions
    for (int kh = 0; kh < kernel_h; ++kh) {
        int out_row = ih * stride_h - pad_h + kh * dilation_h;
        if (out_row < 0 || out_row >= out_h) continue;
        
        for (int kw = 0; kw < kernel_w; ++kw) {
            int out_col = iw * stride_w - pad_w + kw * dilation_w;
            if (out_col < 0 || out_col >= out_w) continue;
            
            int kernel_offset = kh * kernel_w + kw;
            // Loop over output channels in the current group
            for (int oc_offset = 0; oc_offset < out_channels_per_group; ++oc_offset) {
                int oc = group * out_channels_per_group + oc_offset;
                int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) + 
                                   oc_offset * (kernel_h * kernel_w) + kernel_offset;
                float w_val = weight[weight_index];
                float contrib = x_val * w_val;
                
                int out_index = n * (groups * out_channels_per_group * out_h * out_w) +
                                oc * (out_h * out_w) +
                                out_row * out_w + out_col;
                atomicAdd(&output[out_index], contrib);
            }
        }
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
    
    // Ensure contiguous tensors
    x = x.contiguous();
    weight = weight.contiguous();
    
    if (!bias.has_value() || !bias.value().defined())
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    else
        bias = bias.value().contiguous();
    
    // Input dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    
    // Weight dimensions
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    
    // Convolution params
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    // Compute output dimensions
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
    
    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());
    
    // Launch initialize_output_kernel with 3D grid indexing
    dim3 blockInit(16, 16, 1);
    dim3 gridInit((out_w + blockInit.x - 1) / blockInit.x,
                  (out_h + blockInit.y - 1) / blockInit.y,
                  batch * out_channels);
    initialize_output_kernel<<<gridInit, blockInit>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch, out_channels, out_h, out_w);
    
    // Launch conv_transposed2d_scatter_atomic_kernel with 3D grid indexing
    dim3 blockScatter(16, 16, 1);
    dim3 gridScatter((in_w + blockScatter.x - 1) / blockScatter.x,
                     (in_h + blockScatter.y - 1) / blockScatter.y,
                     batch * in_channels);
    conv_transposed2d_scatter_atomic_kernel<<<gridScatter, blockScatter>>>(
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
        out_w);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with 3D Grid Indexing (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
