#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define KERNEL_SIZE 3
#define WARP_SIZE 32
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_kernel(
    const float4* input,
    const float* weight,
    const float* bias,
    float4* output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    __shared__ float shared_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int b = blockIdx.z / out_channels;
    const int oc = blockIdx.z % out_channels;
    
    float sum = bias ? bias[oc] : 0.0f;
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    for (int ic = 0; ic < in_channels; ic++) {
        if (tid < KERNEL_SIZE * KERNEL_SIZE) {
            int wx = tid % KERNEL_SIZE;
            int wy = tid / KERNEL_SIZE;
            shared_weight[wy][wx] = weight[((oc * in_channels + ic) * KERNEL_SIZE + wy) * KERNEL_SIZE + wx];
        }
        __syncthreads();
        
        int in_x_base = out_x * stride - padding;
        int in_y = out_y * stride - padding;
        
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            int in_y_offset = in_y + ky;
            if (in_y_offset >= 0 && in_y_offset < in_height) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int in_x = in_x_base + kx;
                    if (in_x >= 0 && in_x < in_width) {
                        float4 in_val = input[((b * in_channels + ic) * in_height + in_y_offset) * (in_width/4) + (in_x/4)];
                        float weight_val = shared_weight[ky][kx];
                        
                        sum += in_val.x * weight_val;
                        if (in_x + 1 < in_width) sum += in_val.y * weight_val;
                        if (in_x + 2 < in_width) sum += in_val.z * weight_val;
                        if (in_x + 3 < in_width) sum += in_val.w * weight_val;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (out_x < out_width && out_y < out_height) {
        float4 out_val;
        out_val.x = sum;
        out_val.y = sum;
        out_val.z = sum;
        out_val.w = sum;
        output[((b * out_channels + oc) * out_height + out_y) * (out_width/4) + (out_x/4)] = out_val;
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
    
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    
    auto batch_size = x.size(0);
    auto in_channels = x.size(1);
    auto in_height = x.size(2);
    auto in_width = x.size(3);
    auto out_channels = weight.size(0);
    auto out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution");
}