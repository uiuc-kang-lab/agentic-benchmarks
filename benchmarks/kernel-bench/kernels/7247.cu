#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// This kernel implements 2D convolution using a tiled approach with shared memory.
// It minimizes thread synchronizations by calling __syncthreads() only after loading shared memory
// and after the computation phase, ensuring safe re-use of shared memory between iterations over
// input channels.

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int N,              // batch size
    int C,              // total input channels
    int H, int W,       // input height and width
    int out_C, int out_H, int out_W,  // output channels, height and width
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Determine batch index and output channel from blockIdx.z
    int b = blockIdx.z / out_C;
    int oc = blockIdx.z % out_C;

    // Determine the input channels per group and the group id
    int in_channels_per_group = C / groups;
    int group = oc / (out_C / groups);

    // Each thread computes one output pixel
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= out_H || out_x >= out_W) return;

    // Compute the starting coordinate (in input space) for the whole block's tile
    // The top-left output of this block corresponds to input coordinate:
    // tile_origin = (blockIdx * blockDim * stride - padding)
    // We need to adjust the tile origin to account for the block's position
    int tile_origin_x = (blockIdx.x * blockDim.x) * stride - padding;
    int tile_origin_y = (blockIdx.y * blockDim.y) * stride - padding;

    // Compute the dimensions of the input tile needed to cover the block's output region.
    // Since each output pixel uses a kernel window of size kernel_size with dilation,
    // the tile width and height become:
    int tile_width = blockDim.x * stride + (kernel_size - 1) * dilation;
    int tile_height = blockDim.y * stride + (kernel_size - 1) * dilation;

    // Declare dynamic shared memory
    extern __shared__ float sdata[]; // size: tile_width * tile_height floats

    float sum = 0.0f;

    // Loop over the input channels for this group
    for (int c = 0; c < in_channels_per_group; ++c) {
        int ic = group * in_channels_per_group + c;

        // Load the required tile for input[b, ic] into shared memory
        int num_elements = tile_width * tile_height;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid; idx < num_elements; idx += blockDim.x * blockDim.y) {
            int tile_y = idx / tile_width;
            int tile_x = idx % tile_width;
            int in_x = tile_origin_x + tile_x;
            int in_y = tile_origin_y + tile_y;
            if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                sdata[idx] = input[((b * C + ic) * H + in_y) * W + in_x];
            } else {
                sdata[idx] = 0.0f;
            }
        }
        __syncthreads();  // Ensure all threads have loaded the shared tile

        // Compute the starting offset in shared memory corresponding to this thread's output pixel
        // Each thread's output pixel gets its patch from shared memory starting at (threadIdx.y * stride, threadIdx.x * stride)
        int local_origin_y = threadIdx.y * stride;
        int local_origin_x = threadIdx.x * stride;

        // Apply the convolution kernel window
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int shared_y = local_origin_y + ky * dilation;
                int shared_x = local_origin_x + kx * dilation;
                if (shared_y < tile_height && shared_x < tile_width) {
                    float val = sdata[shared_y * tile_width + shared_x];
                    // Weight layout: weight[oc, c, ky, kx] with contiguous layout
                    int weight_idx = ((oc * in_channels_per_group + c) * kernel_size + ky) * kernel_size + kx;
                    float w = weight[weight_idx];
                    sum += val * w;
                }
            }
        }
        __syncthreads();  // Synchronize before loading the next channel's tile into shared memory
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // Write the result to the output tensor. Output layout: [N, out_C, out_H, out_W]
    int out_idx = ((b * out_C + oc) * out_H + out_y) * out_W + out_x;
    output[out_idx] = sum;
}


// Host function that sets up kernel launch parameters and invokes the CUDA kernel.
// It recreates the correct convolution output dimensions and launches a grid over [out_W, out_H] tiles.

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

    // x shape: [N, C, H, W]
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // weight shape: [out_C, C/groups, kernel_size, kernel_size]
    int out_C = weight.size(0);
    int kernel_size = weight.size(2);

    // Compute output dimensions based on convolution formula
    int out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({N, out_C, out_H, out_W}, x.options());

    // Define block dimensions for spatial tiling
    dim3 block(16, 16);
    // Grid: each block covers a tile of the output
    dim3 grid((out_W + block.x - 1) / block.x,
              (out_H + block.y - 1) / block.y,
              N * out_C);

    // Compute the amount of shared memory required per block
    int tile_width = block.x * stride + (kernel_size - 1) * dilation;
    int tile_height = block.y * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_width * tile_height * sizeof(float);

    // Launch the kernel
    conv2d_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W,
        out_C, out_H, out_W,
        kernel_size,
        stride,
        padding,
        dilation,
        groups);

    cudaDeviceSynchronize();
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fast CUDA forward function for 2D convolution with minimal synchronization");
}
