#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel implements 2D convolution with support for groups and optional bias.
// It utilizes dynamic shared memory to cache the kernel weights for each input channel iteration,
// reducing redundant global memory loads. When the overall workload is large, the custom kernel
// will be used; otherwise, the efficient torch::conv2d is called as a fallback.

__global__ void conv2d_combined_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr if not provided
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Compute output spatial coordinates
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // output width index
    int h = blockIdx.y * blockDim.y + threadIdx.y;  // output height index
    int oc = blockIdx.z;  // output channel index

    if (w < out_width && h < out_height && oc < out_channels) {
      // Loop over batches
      for (int b = 0; b < batch_size; ++b) {
        float sum = 0.0f;

        // For grouped convolution, determine the input channel group
        int group_out_channels = out_channels / groups;
        int group = oc / group_out_channels;
        int in_channels_per_group = in_channels / groups;

        // Loop over input channels in the group
        for (int c = 0; c < in_channels_per_group; ++c) {
          int input_channel = group * in_channels_per_group + c;

          // Declare dynamic shared memory for caching the weight tile
          extern __shared__ float weight_shared[];  // size: kernel_height * kernel_width floats
          int kernel_elems = kernel_height * kernel_width;
          int tid = threadIdx.y * blockDim.x + threadIdx.x;
          
          // Each thread loads part of the kernel weight for this channel
          for (int i = tid; i < kernel_elems; i += blockDim.x * blockDim.y) {
            int kh = i / kernel_width;
            int kw = i % kernel_width;
            int weight_idx = (((oc * in_channels_per_group + c) * kernel_height) + kh) * kernel_width + kw;
            weight_shared[i] = weight[weight_idx];
          }
          __syncthreads();

          // Convolve over the kernel window using the cached weights
          for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
              int in_y = h * stride - padding + kh * dilation;
              int in_x = w * stride - padding + kw * dilation;
              if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((b * in_channels + input_channel) * in_height + in_y) * in_width + in_x;
                int weight_offset = kh * kernel_width + kw;
                sum += input[input_idx] * weight_shared[weight_offset];
              }
            }
          }
          __syncthreads();  // Ensure shared memory is ready for next iteration
        }
        
        // Add bias if available
        if (bias != nullptr) {
          sum += bias[oc];
        }
        
        int output_idx = ((b * out_channels + oc) * out_height + h) * out_width + w;
        output[output_idx] = sum;
      }
    }
}

// Forward function: selects the optimal implementation based on the problem size.
// For larger workloads, the custom kernel with shared memory weight caching is used.
// For smaller problems, we fallback to the highly optimized torch::conv2d implementation.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    // Compute output spatial dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    
    auto options = x.options();
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, options);
    
    // Determine the total number of output elements; use custom kernel for larger problems
    int total_output = batch_size * out_channels * out_height * out_width;
    if (total_output > 1024) {  // Threshold can be tuned
        const float* input_ptr = x.data_ptr<float>();
        const float* weight_ptr = weight.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        const float* bias_ptr = (bias.has_value()) ? bias.value().data_ptr<float>() : nullptr;
        
        // Configure the kernel launch parameters. Here, a 16x16 spatial block is used, and each block
        // is responsible for one output channel.
        dim3 block_size(16, 16, 1);
        dim3 grid_size(
            (out_width + block_size.x - 1) / block_size.x,
            (out_height + block_size.y - 1) / block_size.y,
            out_channels
        );
        // Allocate shared memory for one kernel weight tile
        size_t shared_mem_size = kernel_height * kernel_width * sizeof(float);
        
        conv2d_combined_kernel<<<grid_size, block_size, shared_mem_size>>>(
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
            out_height,
            out_width,
            stride,
            padding,
            dilation,
            groups
        );
        
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    } else {
        // For smaller workloads, fallback to PyTorch's optimized implementation
        output = torch::conv2d(
            x,
            weight,
            bias.has_value() ? bias.value() : torch::Tensor(),
            {stride, stride},
            {padding, padding},
            {dilation, dilation},
            groups
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Efficient CUDA 2D Convolution with Shared Memory Weight Caching");
}
