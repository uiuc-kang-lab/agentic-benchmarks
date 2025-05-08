#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device helper function to load an input tile into shared memory
__device__ __forceinline__ void load_shared_tile(
    const float* __restrict__ input,
    float* __restrict__ shared_tile,
    int n,
    int ic,
    int in_channels,
    int in_height,
    int in_width,
    int tile_width,
    int tile_height,
    int in_origin_x,
    int in_origin_y) {

    int total_elements = tile_width * tile_height;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    for (int i = tid; i < total_elements; i += block_size) {
        int tx = i % tile_width;
        int ty = i / tile_width;
        int global_x = in_origin_x + tx;
        int global_y = in_origin_y + ty;
        if (global_x >= 0 && global_x < in_width && global_y >= 0 && global_y < in_height) {
            int input_index = n * (in_channels * in_height * in_width)
                            + ic * (in_height * in_width)
                            + global_y * in_width + global_x;
            shared_tile[i] = input[input_index];
        } else {
            shared_tile[i] = 0.0f;
        }
    }
}

// Device helper function to compute convolution for a single output pixel using the shared memory tile
__device__ __forceinline__ float compute_tile_convolution(
    const float* __restrict__ shared_tile,
    const float* __restrict__ weight,
    int tile_width,
    int kernel_size,
    int stride,
    int dilation,
    int local_x,
    int local_y) {
    float sum = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < kernel_size; ky++) {
        #pragma unroll
        for (int kx = 0; kx < kernel_size; kx++) {
            int s_y = local_y * stride + ky * dilation;
            int s_x = local_x * stride + kx * dilation;
            int index = s_y * tile_width + s_x;
            float in_val = shared_tile[index];
            float w_val = weight[ky * kernel_size + kx];
            sum += in_val * w_val;
        }
    }
    return sum;
}

// Modular conv2d kernel using device helper functions
__global__ void modular_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    // Define output tile dimensions based on block dimensions
    int tile_out_width = blockDim.x;
    int tile_out_height = blockDim.y;
    // Grid dims: grid.x: number of horizontal tiles, grid.y: vertical tiles, grid.z: batch*out_channels combined
    int tile_idx_x = blockIdx.x;
    int tile_idx_y = blockIdx.y;
    int n_oc = blockIdx.z; 
    int n = n_oc / out_channels;
    int oc = n_oc % out_channels;
    
    // Compute starting output coordinates for this block
    int out_start_x = tile_idx_x * tile_out_width;
    int out_start_y = tile_idx_y * tile_out_height;
    
    // Each thread computes one output pixel in the tile
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int out_x = out_start_x + local_x;
    int out_y = out_start_y + local_y;
    
    // Compute input tile dimensions required for this block
    int tile_in_width = tile_out_width * stride + (kernel_size - 1) * dilation;
    int tile_in_height = tile_out_height * stride + (kernel_size - 1) * dilation;
    
    // Compute input tile origin
    int in_origin_x = out_start_x * stride - padding;
    int in_origin_y = out_start_y * stride - padding;
    
    // Allocate shared memory for one input channel tile
    extern __shared__ float shared_tile[];
    
    float sum = 0.0f;
    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile into shared memory for current input channel
        load_shared_tile(input, shared_tile, n, ic, in_channels, in_height, in_width,
                         tile_in_width, tile_in_height, in_origin_x, in_origin_y);
        __syncthreads();
        
        // Compute convolution for this channel if within output bounds
        if (out_x < out_width && out_y < out_height) {
            const float* weight_ptr = weight + (oc * in_channels + ic) * kernel_size * kernel_size;
            sum += compute_tile_convolution(shared_tile, weight_ptr, tile_in_width,
                                            kernel_size, stride, dilation, local_x, local_y);
        }
        __syncthreads();
    }
    
    // Write output if within valid bounds
    if (out_x < out_width && out_y < out_height) {
        if (bias) {
            sum += bias[oc];
        }
        int output_index = n * (out_channels * out_height * out_width)
                         + oc * (out_height * out_width)
                         + out_y * out_width + out_x;
        output[output_index] = sum;
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
    if (bias.has_value())
        CHECK_INPUT(bias.value());

    // Only support non-grouped convolution in this custom kernel
    TORCH_CHECK(groups == 1, "Only groups==1 supported in modular_conv2d_kernel.");

    int batch = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // assuming square kernel

    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Define tile dimensions (tune these based on experimentation)
    const int TILE_OUT_WIDTH = 8;
    const int TILE_OUT_HEIGHT = 8;

    // Compute grid dimensions: number of tiles in x and y directions; grid.z combines batch and output channel
    int grid_x = (out_width + TILE_OUT_WIDTH - 1) / TILE_OUT_WIDTH;
    int grid_y = (out_height + TILE_OUT_HEIGHT - 1) / TILE_OUT_HEIGHT;
    int grid_z = batch * out_channels;

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_OUT_WIDTH, TILE_OUT_HEIGHT);

    int tile_in_width = TILE_OUT_WIDTH * stride + (kernel_size - 1) * dilation;
    int tile_in_height = TILE_OUT_HEIGHT * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_in_width * tile_in_height * sizeof(float);

    modular_conv2d_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, dilation);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular CUDA conv2d kernel with device helper functions");
}
