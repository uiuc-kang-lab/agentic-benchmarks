#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory for both the weight filter and input tiles.
// Grid dimensions:
//   blockIdx.x -> batch index (b)
//   blockIdx.y -> output channel (oc)
//   blockIdx.z -> tile index for outputs (each tile covers a contiguous block of output positions)
// Block dimension (threads.x) is the tile size (number of output positions computed per block).

__global__ void conv1d_kernel_shared_tile(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels,
    int in_size,
    int out_size,
    int kernel_size,
    int stride,
    int dilation
) {
    // Identify batch and output channel
    int b = blockIdx.x;
    int oc = blockIdx.y;

    // Tile offset (starting output index for this block tile)
    int tile_offset = blockIdx.z * blockDim.x;
    int tid = threadIdx.x;
    int global_o = tile_offset + tid;  // Global output index for this thread

    // Compute effective tile size (may be smaller in the last tile)
    int tile_size_eff = blockDim.x;
    if (tile_offset + tile_size_eff > out_size) {
        tile_size_eff = out_size - tile_offset;
    }
    // Compute the required width of the input tile for this block
    // For an output at position o, the last required input index is: o*stride + (kernel_size-1)*dilation.
    // For the tile, we need to load from the start of the first output to the end of the last output's window.
    int tile_width = (tile_size_eff - 1) * stride + (kernel_size - 1) * dilation + 1;

    // Allocate shared memory:
    // First part: s_weight stores the filter weights for this output channel (size: in_channels * kernel_size).
    // Second part: s_tile is reused to load input tiles (size: tile_width).
    extern __shared__ float shared_mem[];
    float* s_weight = shared_mem; 
    float* s_tile = s_weight + in_channels * kernel_size;

    // Load the weight filter for this output channel into shared memory
    int weight_size = in_channels * kernel_size;
    for (int i = tid; i < weight_size; i += blockDim.x) {
        s_weight[i] = weight[oc * weight_size + i];
    }
    __syncthreads();

    float accum = 0.0f;

    // Loop over each input channel and accumulate contributions
    for (int ic = 0; ic < in_channels; ic++) {
        // Pointer to input x for batch b and channel ic
        const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
        // Starting index in x for this tile is determined by the tile_offset and stride
        int tile_input_start = tile_offset * stride;

        // Each thread loads part of the required input tile into shared memory
        for (int j = tid; j < tile_width; j += blockDim.x) {
            int global_idx = tile_input_start + j;
            s_tile[j] = (global_idx < in_size) ? x_ptr[global_idx] : 0.0f;
        }
        __syncthreads();

        // Only compute if this thread corresponds to a valid output position
        if (global_o < out_size) {
            // For each kernel element, use the preloaded shared memory tile
            // The input index within the tile for output at global_o is: (global_o * stride - tile_input_start) = tid * stride,
            // so for kernel offset k, the index becomes: tid * stride + k * dilation
            for (int k = 0; k < kernel_size; k++) {
                int tile_index = tid * stride + k * dilation;
                float x_val = s_tile[tile_index];
                float w_val = s_weight[ic * kernel_size + k];
                accum += x_val * w_val;
            }
        }
        __syncthreads(); // Prepare for next input channel
    }

    // Add bias if provided and write result to global memory
    if (global_o < out_size) {
        if (bias != nullptr) {
            accum += bias[oc];
        }
        // Output shape is [B, out_channels, out_size]
        output[b * (gridDim.y * out_size) + oc * out_size + global_o] = accum;
    }
}

// Forward function exposed to PyTorch

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

    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());

    // Choose a tile size (number of output positions computed per block)
    int tile_size = 128; // can be tuned based on the problem size and GPU
    int num_tiles = (out_size + tile_size - 1) / tile_size;

    // Grid dimensions: one block per (batch, output_channel, tile of outputs)
    dim3 blocks(B, out_channels, num_tiles);
    dim3 threads(tile_size);

    // Shared memory allocation:
    // s_weight: in_channels * kernel_size floats
    // s_tile: maximum tile width = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1 floats
    int tile_width = (tile_size - 1) * stride + (kernel_size - 1) * dilation + 1;
    int shared_mem_size = (in_channels * kernel_size + tile_width) * sizeof(float);

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv1d_kernel_shared_tile<<<blocks, threads, shared_mem_size>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        in_channels,
        in_size,
        out_size,
        kernel_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward with shared memory tiling (CUDA)");
}
