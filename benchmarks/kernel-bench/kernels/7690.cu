#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for the output (tunable parameters)
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

// CUDA kernel for 3D convolution with shared memory tiling
__global__ void conv3d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation
) {
    // Number of tiles along the output depth dimension
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;
    
    // Decode blockIdx.z to obtain batch index, output channel and depth tile index
    int block_index = blockIdx.z;
    int d_tile = block_index % num_d_tiles;
    int tmp = block_index / num_d_tiles;
    int oc = tmp % out_channels;
    int batch = tmp / out_channels;
    
    // Determine the starting coordinate for this output tile
    int out_d_start = d_tile * TILE_D;
    int out_h_start = blockIdx.y * TILE_H;
    int out_w_start = blockIdx.x * TILE_W;
    
    // Thread's coordinate within the tile
    int tw = threadIdx.x;
    int th = threadIdx.y;
    int td = threadIdx.z;
    
    // Global output coordinates
    int out_d = out_d_start + td;
    int out_h = out_h_start + th;
    int out_w = out_w_start + tw;

    // Determine if the thread corresponds to a valid output element
    bool compute = (out_d < out_depth && out_h < out_height && out_w < out_width);
    float sum = 0.0f;
    
    // Compute the shared memory dimensions for the input patch with padding to avoid bank conflicts
    int shared_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shared_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shared_w = TILE_W * stride + (kernel_w - 1) * dilation;
    // Add padding to avoid bank conflicts (32 banks in shared memory)
    int shared_w_padded = (shared_w + 31) & ~31;  // Round up to multiple of 32
    int shared_input_size = shared_d * shared_h * shared_w_padded;
    int kernel_size = kernel_d * kernel_h * kernel_w;
    
    // Dynamically allocated shared memory: first for input patch then for kernel weights
    extern __shared__ float shared_mem[];
    float* smem_input = shared_mem;             // Size: shared_input_size floats
    float* smem_weight = shared_mem + shared_input_size;  // Size: kernel_size floats
    
    // Compute a linear thread id in the block for loading shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    
    // Loop over all input channels (only groups==1 is supported here)
    for (int ic = 0; ic < in_channels; ic++) {
        // Determine the top-left-front corner in the input corresponding to the output tile
        int in_d_start = out_d_start * stride - padding;
        int in_h_start = out_h_start * stride - padding;
        int in_w_start = out_w_start * stride - padding;
        
        // Load the necessary input patch for current input channel into shared memory
        for (int idx = tid; idx < shared_input_size; idx += block_size) {
            int sd = idx / (shared_h * shared_w);
            int rem = idx % (shared_h * shared_w);
            int sh = rem / shared_w;
            int sw = rem % shared_w;
            int in_d = in_d_start + sd;
            int in_h = in_h_start + sh;
            int in_w = in_w_start + sw;
            float in_val = 0.0f;
            if (in_d >= 0 && in_d < in_depth &&
                in_h >= 0 && in_h < in_height &&
                in_w >= 0 && in_w < in_width) {
                int input_idx = (((batch * in_channels + ic) * in_depth + in_d) * in_height + in_h) * in_width + in_w;
                in_val = input[input_idx];
            }
            smem_input[idx] = in_val;
        }
        __syncthreads();
        
        // Load the weight kernel for the current (oc, ic) pair into shared memory
        for (int idx = tid; idx < kernel_size; idx += block_size) {
            int kd = idx / (kernel_h * kernel_w);
            int rem = idx % (kernel_h * kernel_w);
            int kh = rem / kernel_w;
            int kw = rem % kernel_w;
            int weight_idx = ((((oc * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
            smem_weight[idx] = weight[weight_idx];
        }
        __syncthreads();
        
        // Each thread computes its partial sum for the current input channel
        if (compute) {
            int base_d = td * stride;
            int base_h = th * stride;
            int base_w = tw * stride;
            float channel_sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int kd = k / (kernel_h * kernel_w);
                int rem = k % (kernel_h * kernel_w);
                int kh = rem / kernel_w;
                int kw = rem % kernel_w;
                int sd = base_d + kd * dilation;
                int sh = base_h + kh * dilation;
                int sw = base_w + kw * dilation;
                int input_offset = (sd * shared_h + sh) * shared_w + sw;
                channel_sum += smem_weight[k] * smem_input[input_offset];
            }
            sum += channel_sum;
        }
        __syncthreads();
    } // end loop over input channels
    
    // Write the result to the output tensor, adding bias if provided
    if (compute) {
        int out_idx = (((batch * out_channels + oc) * out_depth + out_d) * out_height + out_h) * out_width + out_w;
        float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;
        output[out_idx] = sum + bias_val;
    }
}

// Host function that prepares and launches the custom kernel
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // This implementation currently supports only groups == 1
    TORCH_CHECK(groups == 1, "shared_memory_conv3d only supports groups == 1");
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(weight.dim() == 5, "Weight must be a 5D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }
    
    // Get dimensions from input and weight tensors
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    
    auto options = input.options();
    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);
    
    // Set up block and grid dimensions for the kernel launch
    dim3 block(TILE_W, TILE_H, TILE_D);
    int grid_x = (out_width + TILE_W - 1) / TILE_W;
    int grid_y = (out_height + TILE_H - 1) / TILE_H;
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;
    int grid_z = batch_size * out_channels * num_d_tiles;
    dim3 grid(grid_x, grid_y, grid_z);
    
    // Compute the required dynamic shared memory size per block
    int shared_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shared_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shared_w = TILE_W * stride + (kernel_w - 1) * dilation;
    int shared_input_size = shared_d * shared_h * shared_w;
    int kernel_size = kernel_d * kernel_h * kernel_w;
    size_t shared_mem_size = (shared_input_size + kernel_size) * sizeof(float);
    
    // Launch the kernel
    conv3d_shared_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels,
        kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with shared memory optimization (CUDA)");
}
