#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void conv2d_warp_aligned_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups) {

    // Calculate global position
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Calculate output position
    const int out_x = blockIdx.x * WARP_SIZE + lane_id;
    const int out_y = (blockIdx.y * blockDim.y + threadIdx.y);
    const int oc = blockIdx.z;

    // Early exit if out of bounds
    if (out_x >= out_width || out_y >= out_height || oc >= out_channels) {
        return;
    }

    // Pre-calculate group-related constants
    const int group_out_channels = out_channels / groups;
    const int group = oc / group_out_channels;
    const int in_channels_per_group = in_channels / groups;
    const int group_in_offset = group * in_channels_per_group;

    // Process all batches
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        
        // Calculate input boundaries for this output position
        const int in_y_start = out_y * stride - padding;
        const int in_x_start = out_x * stride - padding;
        
        // Process input channels in groups
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            const int input_channel = group_in_offset + ic;
            
            // Pre-calculate input batch offset
            const int in_b_offset = (b * in_channels + input_channel) * in_height;
            const int weight_c_offset = ((oc * in_channels_per_group + ic) * kernel_height);

            // Process kernel height
            for (int kh = 0; kh < kernel_height; kh++) {
                const int in_y = in_y_start + kh * dilation;
                
                // Skip if outside vertical bounds
                if (in_y >= 0 && in_y < in_height) {
                    const int in_row_offset = in_b_offset + in_y * in_width;
                    const int weight_row_offset = weight_c_offset + kh * kernel_width;
                    
                    // Process kernel width
                    for (int kw = 0; kw < kernel_width; kw++) {
                        const int in_x = in_x_start + kw * dilation;
                        
                        // Skip if outside horizontal bounds
                        if (in_x >= 0 && in_x < in_width) {
                            sum += input[in_row_offset + in_x] *
                                  weight[weight_row_offset + kw];
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // Write output
        const int out_idx = ((b * out_channels + oc) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Configure grid and block dimensions for warp-aligned processing
    dim3 block_size(WARP_SIZE, 8, 1);  // 32x8=256 threads per block
    dim3 grid_size(
        (out_width + WARP_SIZE - 1) / WARP_SIZE,
        (out_height + block_size.y - 1) / block_size.y,
        out_channels
    );

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv2d_warp_aligned_kernel<<<grid_size, block_size>>>(
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

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA 2D Convolution with Warp-Aligned Processing");
}