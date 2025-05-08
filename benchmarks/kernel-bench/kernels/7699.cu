#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable tile sizes for output spatial dimensions
#define TILE_D 4
#define TILE_H 4
#define TILE_W 4
// Chunk of input channels to load per iteration
#define CH_TILE 4

// Optimized 3D convolution kernel using shared memory tiling over input channels and spatial patch
__global__ void conv3d_shared_kernel_opt(
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
    // Calculate number of depth tiles
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;

    // Decode blockIdx.z to get batch index, output channel, and depth tile index
    int tmp = blockIdx.z;
    int batch = tmp / (out_channels * num_d_tiles);
    tmp = tmp % (out_channels * num_d_tiles);
    int oc = tmp / num_d_tiles;
    int tile_d_idx = tmp % num_d_tiles;

    int out_d_start = tile_d_idx * TILE_D;
    int out_h_start = blockIdx.y * TILE_H;
    int out_w_start = blockIdx.x * TILE_W;

    // Thread's coordinates within the output tile
    int td = threadIdx.z;
    int th = threadIdx.y;
    int tw = threadIdx.x;

    int out_d = out_d_start + td;
    int out_h = out_h_start + th;
    int out_w = out_w_start + tw;

    bool valid_output = (out_d < out_depth) && (out_h < out_height) && (out_w < out_width);
    float sum = 0.0f; float partials[CH_TILE] = {0};

    // Shared memory dimensions for the input patch corresponding to the output tile
    int shm_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shm_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shm_w = TILE_W * stride + (kernel_w - 1) * dilation;
    int shm_patch_size = shm_d * shm_h * shm_w;  // per input channel

    // Dynamic shared memory layout: first for input patch then for weights
    // Allocated size: CH_TILE * (shm_patch_size + kernel_d*kernel_h*kernel_w) floats
    extern __shared__ float shared_mem[];
    float* smem_input = shared_mem; 
    float* smem_weight = shared_mem + CH_TILE * shm_patch_size;

    // Loop over input channels in chunks
    for (int ic_base = 0; ic_base < in_channels; ic_base += CH_TILE) {
        int current_tile = (ic_base + CH_TILE <= in_channels) ? CH_TILE : (in_channels - ic_base);

        // Load input patch for current channel chunk into shared memory
        // Total elements = current_tile * shm_patch_size
        int total_elems = current_tile * shm_patch_size;
        int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int block_size = blockDim.x * blockDim.y * blockDim.z;
        for (int idx = thread_id; idx < total_elems; idx += block_size) {
            int ch = idx / shm_patch_size;
            int rem = idx % shm_patch_size;
            int s_d = rem / (shm_h * shm_w);
            int rem2 = rem % (shm_h * shm_w);
            int s_h = rem2 / shm_w;
            int s_w = rem2 % shm_w;
            // Global coordinates for input
            int in_d = out_d_start * stride - padding + s_d;
            int in_h = out_h_start * stride - padding + s_h;
            int in_w = out_w_start * stride - padding + s_w;
            float val = 0.0f;
            if (in_d >= 0 && in_d < in_depth &&
                in_h >= 0 && in_h < in_height &&
                in_w >= 0 && in_w < in_width) {
                int input_idx = (((batch * in_channels + (ic_base + ch)) * in_depth + in_d) * in_height + in_h) * in_width + in_w;
                val = input[input_idx];
            }
            smem_input[ch * shm_patch_size + rem] = val;
        }
        __syncthreads();

        // Load weight for current channel chunk into shared memory
        int weight_elems = current_tile * kernel_d * kernel_h * kernel_w;
        for (int idx = thread_id; idx < weight_elems; idx += block_size) {
            int ch = idx / (kernel_d * kernel_h * kernel_w);
            int rem = idx % (kernel_d * kernel_h * kernel_w);
            int kd = rem / (kernel_h * kernel_w);
            int rem2 = rem % (kernel_h * kernel_w);
            int kh = rem2 / kernel_w;
            int kw = rem2 % kernel_w;
            int weight_idx = ((((oc * in_channels) + (ic_base + ch)) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
            smem_weight[ch * (kernel_d * kernel_h * kernel_w) + rem] = weight[weight_idx];
        }
        __syncthreads();

        // Compute convolution for this input channel chunk if the output is valid
        if (valid_output) {
            // Base offset in the shared memory patch for the current output pixel
            int base_d = td * stride;
            int base_h = th * stride;
            int base_w = tw * stride;
            for (int c = 0; c < current_tile; c++) {
                float partial = 0.0f;
                int weight_offset = c * (kernel_d * kernel_h * kernel_w);
                int input_offset = c * shm_patch_size;
                // Iterate over the filter window
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int s_d = base_d + kd * dilation;
                            int s_h = base_h + kh * dilation;
                            int s_w = base_w + kw * dilation;
                            int inp_index = input_offset + (s_d * shm_h * shm_w + s_h * shm_w + s_w);
                            int w_index = weight_offset + (kd * kernel_h * kernel_w + kh * kernel_w + kw);
                            partial += smem_weight[w_index] * smem_input[inp_index];
                        }
                    }
                }
                sum += partial;
            }
        }
        __syncthreads(); // Prepare for next input channel chunk
    }

    // Write output if in valid range
    if (valid_output) {
        int out_idx = (((batch * out_channels + oc) * out_depth + out_d) * out_height + out_h) * out_width + out_w;
        float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;
        output[out_idx] = sum + bias_val;
    }
}

// Host function to launch the optimized 3D convolution kernel
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    TORCH_CHECK(groups == 1, "shared_mem_conv3d_opt only supports groups == 1");
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

    // Calculate output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    at::Tensor output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, options);

    // Grid and block dimensions
    int grid_x = (out_width + TILE_W - 1) / TILE_W;
    int grid_y = (out_height + TILE_H - 1) / TILE_H;
    int num_d_tiles = (out_depth + TILE_D - 1) / TILE_D;
    int grid_z = batch_size * out_channels * num_d_tiles;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_W, TILE_H, TILE_D);

    // Shared memory size per block
    int shm_d = TILE_D * stride + (kernel_d - 1) * dilation;
    int shm_h = TILE_H * stride + (kernel_h - 1) * dilation;
    int shm_w = TILE_W * stride + (kernel_w - 1) * dilation;
    int shm_patch_size = shm_d * shm_h * shm_w; 
    int weight_size = kernel_d * kernel_h * kernel_w;
    size_t shared_mem_size = sizeof(float) * (CH_TILE * (shm_patch_size + weight_size));

    conv3d_shared_kernel_opt<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
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
    m.def("forward", &forward, "Optimized 3D convolution forward with shared memory optimization (CUDA)");
}
