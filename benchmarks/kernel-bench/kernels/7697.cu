#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions (tunable parameters)
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4

// Maximum number of floats that can be stored in constant memory for weights
#define MAX_CONST_WEIGHT_SIZE (1024 * 1024)

// Declare constant memory for kernel weights
__constant__ float d_const_weight[MAX_CONST_WEIGHT_SIZE];

// CUDA kernel for 3D convolution using constant memory for weights
__global__ void conv3d_constant_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,  // can be nullptr
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
    // Determine the number of depth tiles in the output
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;
    
    // Decode blockIdx.z to obtain batch index, output channel, and depth tile index
    int block_index = blockIdx.z;
    int d_tile = block_index % num_d_tiles;
    int tmp = block_index / num_d_tiles;
    int oc = tmp % out_channels;
    int batch = tmp / out_channels;
    
    // Starting coordinates for this output tile
    int out_d_start = d_tile * TILE_D;
    int out_h_start = blockIdx.y * TILE_H;
    int out_w_start = blockIdx.x * TILE_W;

    // Thread's coordinate within the tile
    int td = threadIdx.z;
    int th = threadIdx.y;
    int tw = threadIdx.x;

    // Global output coordinates
    int out_d = out_d_start + td;
    int out_h = out_h_start + th;
    int out_w = out_w_start + tw;
    bool compute = (out_d < out_depth && out_h < out_height && out_w < out_width);
    float sum = 0.0f;

    // Calculate shared memory dimensions for the input patch
    int shared_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shared_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shared_w = TILE_W * stride + (kernel_w - 1) * dilation;
    int shared_input_size = shared_d * shared_h * shared_w;

    extern __shared__ float smem_input[];  // dynamically allocated shared memory for input patch

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int block_size = blockDim.x * blockDim.y * blockDim.z;

    // Loop over all input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Starting coordinate in input corresponding to this output tile
        int in_d_start = out_d_start * stride - padding;
        int in_h_start = out_h_start * stride - padding;
        int in_w_start = out_w_start * stride - padding;
        
        // Load the required input patch into shared memory
        for (int idx = tid; idx < shared_input_size; idx += block_size) {
            int sd = idx / (shared_h * shared_w);
            int rem = idx % (shared_h * shared_w);
            int sh = rem / shared_w;
            int sw = rem % shared_w;
            int in_d = in_d_start + sd;
            int in_h = in_h_start + sh;
            int in_w = in_w_start + sw;
            float val = 0.0f;
            if (in_d >= 0 && in_d < in_depth &&
                in_h >= 0 && in_h < in_height &&
                in_w >= 0 && in_w < in_width) {
                int input_idx = (((batch * in_channels + ic) * in_depth + in_d) * in_height + in_h) * in_width + in_w;
                val = input[input_idx];
            }
            smem_input[idx] = val;
        }
        __syncthreads();
        
        // Perform convolution for the current input channel using weights from constant memory
        if (compute) {
            int base_d = td * stride;
            int base_h = th * stride;
            int base_w = tw * stride;
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int sd = base_d + kd * dilation;
                        int sh = base_h + kh * dilation;
                        int sw = base_w + kw * dilation;
                        int smem_idx = (sd * shared_h + sh) * shared_w + sw;
                        int weight_idx = ((((oc * in_channels) + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += smem_input[smem_idx] * d_const_weight[weight_idx];
                    }
                }
            }
        }
        __syncthreads();
    } // end loop over input channels
    
    // Write the computed value to the output tensor, adding bias if provided
    if (compute) {
        int out_idx = (((batch * out_channels + oc) * out_depth + out_d) * out_height + out_h) * out_width + out_w;
        float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;
        output[out_idx] = sum + bias_val;
    }
}

// Host function that prepares and launches the kernel
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "constant_memory_conv3d only supports groups == 1");
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.dim() == 5, "Input must be a 5D tensor");
    TORCH_CHECK(weight.dim() == 5, "Weight must be a 5D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Copy weight tensor into constant memory. Ensure it fits within the hardware limits.
    size_t weight_numel = weight.numel();
    TORCH_CHECK(weight_numel <= MAX_CONST_WEIGHT_SIZE, "Weight tensor exceeds constant memory size");
    cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), weight_numel * sizeof(float));

    // Set up grid and block dimensions
    dim3 block(TILE_W, TILE_H, TILE_D);
    int grid_x = (out_width + TILE_W - 1) / TILE_W;
    int grid_y = (out_height + TILE_H - 1) / TILE_H;
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;
    int grid_z = batch_size * out_channels * num_d_tiles;
    dim3 grid(grid_x, grid_y, grid_z);

    // Compute dynamic shared memory size required for the input patch
    int shared_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shared_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shared_w = TILE_W * stride + (kernel_w - 1) * dilation;
    int shared_input_size = shared_d * shared_h * shared_w;
    size_t shared_mem_size = shared_input_size * sizeof(float);

    conv3d_constant_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        out_depth, out_height, out_width,
        stride, padding, dilation
    );

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward with constant memory optimization (CUDA)");
}
