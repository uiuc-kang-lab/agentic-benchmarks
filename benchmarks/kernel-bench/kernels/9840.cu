#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024

__global__ void depthwise_conv2d_gridstride_shared_kernel(
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
    int channels_per_group,
    int total_elements
) {
    extern __shared__ float shared_weight[];
    
    // Load weights into shared memory
    int tid = threadIdx.x;
    int oc = blockIdx.y;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;
    
    // Cooperatively load weights for this output channel
    int weight_size = kernel_size * kernel_size;
    for (int i = tid; i < weight_size; i += blockDim.x) {
        shared_weight[i] = weight[in_ch * (channels_per_group * weight_size) +
                                weight_ch * weight_size + i];
    }
    __syncthreads();
    
    // Calculate grid-stride loop parameters
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements using grid-stride loop
    for (int idx = gid; idx < total_elements; idx += stride) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int c_out = (idx / (output_w * output_h)) % out_channels;
        int b = idx / (output_w * output_h * out_channels);
        
        // Only process if this thread's channel matches the block's channel
        if (c_out == oc && b < batch_size) {
            float sum = 0.0f;
            
            #pragma unroll 3
            for (int ky = 0; ky < kernel_size; ++ky) {
                int h_in = h_out * stride - padding + ky;
                
                if (h_in >= 0 && h_in < input_h) {
                    #pragma unroll 3
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int w_in = w_out * stride - padding + kx;
                        
                        if (w_in >= 0 && w_in < input_w) {
                            int input_idx = b * (in_channels * input_h * input_w) +
                                          in_ch * (input_h * input_w) +
                                          h_in * input_w + w_in;
                            
                            sum += input[input_idx] * shared_weight[ky * kernel_size + kx];
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            
            output[idx] = sum;
        }
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
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;
    
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    int total_elements = batch_size * out_channels * output_h * output_w;
    int elements_per_channel = total_elements / out_channels;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((elements_per_channel + BLOCK_SIZE - 1) / BLOCK_SIZE, out_channels);
    
    size_t shared_mem_size = kernel_size * kernel_size * sizeof(float);
    
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    
    depthwise_conv2d_gridstride_shared_kernel<<<grid, block, shared_mem_size>>>(
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
        channels_per_group,
        total_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with Grid-Stride and Shared Memory (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}