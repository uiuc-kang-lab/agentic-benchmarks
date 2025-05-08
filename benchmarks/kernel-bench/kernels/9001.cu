#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// This kernel divides the output tensor's width into tiles and assigns each (batch, out_channel) pair
// to a block. Threads within each block collaboratively compute the output for a tile, ensuring
// even workload distribution across threads and blocks.
__global__ void conv1d_kernel_tiled(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    // Each block is responsible for computing a tile of the output for one (batch, out_channel) pair.
    // blockIdx.y encodes the linear index for the (b, oc) pair.
    int linear = blockIdx.y;
    int b = linear / out_channels;
    int oc = linear % out_channels;

    // Each block handles a tile of the out_size dimension
    int tile_start = blockIdx.x * TILE_SIZE;
    int tile_end = tile_start + TILE_SIZE;
    if (tile_end > out_size) tile_end = out_size;

    // Loop over the tile with a stride of blockDim.x so that threads evenly cover the tile.
    for (int i = tile_start + threadIdx.x; i < tile_end; i += blockDim.x) {
        float sum = 0.0f;
        int input_offset = i * stride;  // starting index in the input for this output position
        int x_base = b * (in_channels * in_size);
        int w_base = oc * (in_channels * kernel_size);
        
        // Perform the convolution over all input channels and kernel elements.
        for (int ic = 0; ic < in_channels; ++ic) {
            int x_offset = x_base + ic * in_size;
            int w_offset = w_base + ic * kernel_size;
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                // The output size is computed to ensure the convolution window is fully valid,
                // so no boundary check is needed here.
                sum += x[x_offset + input_offset + k * dilation] * weight[w_offset + k];
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[b * (out_channels * out_size) + oc * out_size + i] = sum;
    }
}

// Forward function exposed via pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "Bias size mismatch");
    }
    
    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Compute output size ensuring the convolution window is valid for every output element
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");
    
    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;
    
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();
    
    // Grid configuration:
    //   - gridDim.x covers the out_size dimension in tiles of TILE_SIZE
    //   - gridDim.y covers all (batch, out_channel) pairs
    dim3 blocks((out_size + TILE_SIZE - 1) / TILE_SIZE, B * out_channels);
    int threads = 32;  // A warp size of 32 ensures even workload distribution within each tile.
    
    conv1d_kernel_tiled<<<blocks, threads>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with tiled workload distribution");
}
