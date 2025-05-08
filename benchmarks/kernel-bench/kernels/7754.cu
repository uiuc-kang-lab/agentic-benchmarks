#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions for output spatial block
#define TILE_W 8
#define TILE_H 8
#define TILE_D 8

// conv3d_shared_kernel: Each block computes one tile of output for a given batch and output channel.
// The kernel caches the weight filter in shared memory using minimal synchronization (__syncthreads() only after loading).

template <typename scalar_t>
__global__ void conv3d_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // may be NULL if not provided
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels, int in_d, int in_h, int in_w,
    int out_channels, int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int groups) {

    // Compute spatial tiling for output
    int num_tiles_w = (out_w + TILE_W - 1) / TILE_W;
    int num_tiles_h = (out_h + TILE_H - 1) / TILE_H;
    int num_tiles_d = (out_d + TILE_D - 1) / TILE_D;
    int tile_index = blockIdx.x; // flatten index for spatial tile
    int tile_w_offset = (tile_index % num_tiles_w) * TILE_W;
    int tile_h_offset = ((tile_index / num_tiles_w) % num_tiles_h) * TILE_H;
    int tile_d_offset = (tile_index / (num_tiles_w * num_tiles_h)) * TILE_D;

    // Identify the batch and output channel for this block
    int b = blockIdx.z;         // batch index
    int oc = blockIdx.y;        // output channel index

    // For grouped convolution
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups; // not directly used but for clarity
    int group = oc / out_channels_per_group;  
    int in_channel_base = group * in_channels_per_group;

    // Allocate shared memory for the filter weights for this output channel
    // Filter dimensions: [in_channels_per_group, kernel_d, kernel_h, kernel_w]
    extern __shared__ char smem[];
    scalar_t* s_weight = reinterpret_cast<scalar_t*>(smem);
    int filter_size = in_channels_per_group * kernel_d * kernel_h * kernel_w;

    // Cooperative loading of the filter for this output channel into shared memory
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int i = tid; i < filter_size; i += block_size) {
        s_weight[i] = weight[oc * filter_size + i];
    }
    __syncthreads();  // synchronize once after loading weights

    // Each thread computes one output element in the tile
    int ow = tile_w_offset + threadIdx.x;
    int oh = tile_h_offset + threadIdx.y;
    int od = tile_d_offset + threadIdx.z;

    if (ow < out_w && oh < out_h && od < out_d) {
        scalar_t sum = 0;
        // Loop over input channels in the group and kernel spatial dimensions
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            int current_in_channel = in_channel_base + ic;
            for (int kd = 0; kd < kernel_d; ++kd) {
                int id = od * stride - padding + kd * dilation;
                if (id < 0 || id >= in_d) continue;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    int ih = oh * stride - padding + kh * dilation;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int iw = ow * stride - padding + kw * dilation;
                        if (iw < 0 || iw >= in_w) continue;
                        int input_idx = (((b * in_channels + current_in_channel) * in_d + id) * in_h + ih) * in_w + iw;
                        int w_idx = ((ic * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * s_weight[w_idx];
                    }
                }
            }
        }
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int output_idx = (((b * out_channels + oc) * out_d + od) * out_h + oh) * out_w + ow;
        output[output_idx] = sum;
    }
}

// Host function: sets up grid/block dimensions and launches the kernel
at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Weight dimensions and output channels
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Calculate output dimensions using standard convolution formula
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    // Compute grid dimensions for spatial tiling
    int num_tiles_w = (out_w + TILE_W - 1) / TILE_W;
    int num_tiles_h = (out_h + TILE_H - 1) / TILE_H;
    int num_tiles_d = (out_d + TILE_D - 1) / TILE_D;
    int spatial_tiles = num_tiles_w * num_tiles_h * num_tiles_d;
    
    // Grid: x dimension for spatial tiles, y for output channels, z for batch
    dim3 blockDim(TILE_W, TILE_H, TILE_D);
    dim3 gridDim(spatial_tiles, out_channels, batch_size);

    int in_channels_per_group = in_channels / groups;
    int filter_size = in_channels_per_group * kernel_d * kernel_h * kernel_w;
    size_t shared_mem_size = filter_size * sizeof(float);  // works for float/double based on AT_DISPATCH

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_shared_cuda", ([&] {
        const auto* input_ptr = input.data_ptr<scalar_t>();
        const auto* weight_ptr = weight.data_ptr<scalar_t>();
        const auto* bias_ptr = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
        scalar_t* output_ptr = output.data_ptr<scalar_t>();

        conv3d_shared_kernel<scalar_t><<<gridDim, blockDim, shared_mem_size>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size,
            in_channels, in_d, in_h, in_w,
            out_channels, out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride, padding, dilation,
            groups);
        cudaDeviceSynchronize();
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward CUDA kernel with shared memory for weights");
}
