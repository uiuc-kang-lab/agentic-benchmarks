#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define KERNEL_SIZE 3
#define MAX_WEIGHT_SIZE 16384  // Maximum number of floats to store in constant memory (~64KB)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for the kernel weights
__constant__ float d_weight_const[MAX_WEIGHT_SIZE];

// CUDA kernel that reads convolution weights from constant memory
__global__ void conv2d_constmem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation) {

    // Determine thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;  // now represents tile index
    int by = blockIdx.y;
    
    // Decode batch and output channel from grid dimension
    int b_oc = blockIdx.z;
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;
    
    // Compute output indices for this thread
    int out_h = by * TILE_SIZE + ty;
    int out_w = bx * TILE_SIZE + tx;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Dimensions of shared memory tile for input channel
    // It covers the output tile scaled by the stride plus the additional border required by the kernel
    int smem_height = TILE_SIZE * stride + (KERNEL_SIZE - 1) * dilation;
    int smem_width  = TILE_SIZE * stride + (KERNEL_SIZE - 1) * dilation;
    
    // Top-left coordinate in the input feature map for this block's shared memory tile
    int in_row_origin = by * TILE_SIZE * stride - padding;
    int in_col_origin = bx * TILE_SIZE * stride - padding;

    // Declare dynamic shared memory buffer for the input tile
    extern __shared__ float smem[];

    // Loop over each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // Pointer to the current input channel for batch b
        const float* in_ptr = input + ((b * in_channels + ic) * in_height * in_width);
        
        // Calculate total number of elements to load into shared memory
        int smem_size = smem_height * smem_width;
        
        // Each thread cooperatively loads elements into shared memory
        // Compute a linear index for the shared memory tile
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < smem_size; idx += blockDim.x * blockDim.y) {
            int i = idx / smem_width;
            int j = idx % smem_width;
            int global_i = in_row_origin + i;
            int global_j = in_col_origin + j;
            if (global_i >= 0 && global_i < in_height && global_j >= 0 && global_j < in_width) {
                smem[idx] = in_ptr[global_i * in_width + global_j];
            } else {
                smem[idx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Only process valid output indices
        if (out_h < out_height && out_w < out_width) {
            // Compute the starting location in shared memory corresponding to the current output element
            int local_row = ty * stride;
            int local_col = tx * stride;
            
            // Apply the convolution kernel over the loaded tile
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    int smem_r = local_row + kh * dilation;
                    int smem_c = local_col + kw * dilation;
                    int smem_idx = smem_r * smem_width + smem_c;
                    int weight_idx = ((oc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
                    sum += smem[smem_idx] * d_weight_const[weight_idx];
                }
            }
        }
        __syncthreads();  // Prepare for next input channel
    }
    
    if (out_h < out_height && out_w < out_width) {
        int output_idx = ((b * out_channels + oc) * out_height + out_h) * out_width + out_w;
        output[output_idx] = sum;
    }
}

// Host function to copy weights to constant memory and launch the kernel
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

    // Retrieve tensor dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    
    int kernel_size = weight.size(2);
    TORCH_CHECK(kernel_size == KERNEL_SIZE, "Kernel size mismatch.");

    int out_height = (in_height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());
    
    // Ensure the weight tensor fits in constant memory
    int weight_elements = weight.numel();
    TORCH_CHECK(weight_elements <= MAX_WEIGHT_SIZE, "Weight tensor is too large for constant memory.");
    
    // Copy weights from the input tensor to constant memory
    cudaMemcpyToSymbol(d_weight_const, weight.data_ptr<float>(), weight_elements * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_constmem_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        stride,
        padding,
        dilation
    );

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution using constant memory for weights");
}
