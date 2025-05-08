#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3  // Assuming 3x3 kernel for shared memory optimization
#define SHARED_SIZE (BLOCK_SIZE + KERNEL_SIZE - 1)

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding) {
    
    __shared__ float shared_input[SHARED_SIZE][SHARED_SIZE];
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int b = blockIdx.z;
    
    const int x = bx + tx;
    const int y = by + ty;
    
    // Process each output channel
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        
        // Process each input channel
        for (int ic = 0; ic < in_channels; ++ic) {
            // Load kernel weights into shared memory
            if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
                int weight_idx = ((oc * in_channels + ic) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx;
                shared_weight[ty][tx] = weight[weight_idx];
            }
            __syncthreads();
            
            // Load input tile into shared memory
            for (int i = ty; i < SHARED_SIZE; i += BLOCK_SIZE) {
                for (int j = tx; j < SHARED_SIZE; j += BLOCK_SIZE) {
                    int ih = by + i - padding;
                    int iw = bx + j - padding;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        shared_input[i][j] = input[((b * in_channels + ic) * input_height + ih) * input_width + iw];
                    } else {
                        shared_input[i][j] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            // Compute convolution for this input channel
            if (x < output_width && y < output_height) {
                for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
                    for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                        int sy = ty * stride + ki;
                        int sx = tx * stride + kj;
                        sum += shared_input[sy][sx] * shared_weight[ki][kj];
                    }
                }
            }
            __syncthreads();
        }
        
        // Write output
        if (x < output_width && y < output_height) {
            int output_idx = ((b * out_channels + oc) * output_height + y) * output_width + x;
            output[output_idx] = sum;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.size(2) == KERNEL_SIZE && weight.size(3) == KERNEL_SIZE, 
               "This implementation assumes 3x3 kernels");
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto input_height = x.size(2);
    auto input_width = x.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    auto output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                             x.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                batch_size);
    
    conv2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding);
    
    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory optimization");
}