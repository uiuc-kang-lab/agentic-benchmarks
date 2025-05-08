#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile width for the output dimension computed per block
// Adjust TILE_WIDTH based on typical workload, here we use 128
#define TILE_WIDTH 128

// This kernel uses shared memory to load both the required input tile and the filter weights
// for a (batch, output channel, output tile) block. The grid is organized as:
//   grid.x: number of tiles in the output spatial dimension,
//   grid.y: output channels,
//   grid.z: batch size.
// Each block loads a tile of input data and the complete filter for the given output channel
// into shared memory, then computes a tile of outputs. The required input tile length is computed
// as: (tile_width - 1) * stride + 1 + (kernel_size - 1) * dilation.

__global__ void conv1d_kernel_shared(
    const float* __restrict__ x,       // [B, in_channels, in_size]
    const float* __restrict__ weight,  // [out_channels, in_channels, kernel_size]
    const float* __restrict__ bias,    // [out_channels]
    float* __restrict__ output,        // [B, out_channels, out_size]
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation,
    int input_tile_len_max           // maximum input tile length (dependent on TILE_WIDTH)
) {
    // Determine block indices
    // grid.x: tile index along the output spatial dimension
    // grid.y: output channel
    // grid.z: batch index
    int tile_idx = blockIdx.x;            // tile index along out dimension
    int oc = blockIdx.y;                  // output channel
    int b = blockIdx.z;                   // batch index

    // Compute starting output index for this tile
    int out_start = tile_idx * TILE_WIDTH;
    // Compute the actual tile width (may be smaller at the boundary)
    int current_tile_width = (out_start + TILE_WIDTH <= out_size) ? TILE_WIDTH : (out_size - out_start);

    // Compute the required input tile length for this block
    // For each output element, we need an input window of size: 1 + (kernel_size - 1)*dilation, and
    // between adjacent outputs the starting index increases by 'stride'.
    int input_tile_len = (current_tile_width - 1) * stride + 1 + (kernel_size - 1) * dilation;
    // Note: We allocate shared memory based on the maximum tile size (TILE_WIDTH)

    // Dynamic shared memory allocation:
    // [0, in_channels * input_tile_len_max) for input tile
    // [in_channels * input_tile_len_max, in_channels * input_tile_len_max + in_channels * kernel_size) for weight
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem; // shared input tile
    float* s_weight = s_input + in_channels * input_tile_len_max; // shared filter weights

    // Load filter weights for this output channel into shared memory
    int weight_elements = in_channels * kernel_size;
    for (int index = threadIdx.x; index < weight_elements; index += blockDim.x) {
        s_weight[index] = weight[oc * weight_elements + index];
    }

    // Load input tile for this block from global memory
    // For the given batch 'b', for each input channel, we load elements starting from index = out_start * stride
    int input_elements = in_channels * input_tile_len;  // actual number of elements needed for this tile
    for (int index = threadIdx.x; index < input_elements; index += blockDim.x) {
        int ic = index / input_tile_len;
        int offset = index % input_tile_len;
        int global_index = out_start * stride + offset; // index in the input signal
        int base = b * (in_channels * in_size) + ic * in_size;
        s_input[index] = (global_index < in_size) ? x[base + global_index] : 0.0f;
    }

    __syncthreads();

    // Each thread computes one output element in the tile if within current_tile_width
    if (threadIdx.x < current_tile_width) {
        int o = out_start + threadIdx.x;  // actual output index in the full output
        float sum = 0.0f;
        // For each input channel
        for (int ic = 0; ic < in_channels; ++ic) {
            // For each kernel position
            for (int k = 0; k < kernel_size; ++k) {
                int s_index = threadIdx.x * stride + k * dilation;  // position in shared input
                // Access shared memory: use the maximum input tile length for indexing
                sum += s_input[ic * input_tile_len_max + s_index] * s_weight[ic * kernel_size + k];
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        // Write the computed output: output[b, oc, o]
        output[b * (out_channels * out_size) + oc * out_size + o] = sum;
    }
}

// Forward function called via pybind11
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
    if (output.numel() == 0) return output;

    // Set tile width based on our defined TILE_WIDTH constant
    int tile_width = TILE_WIDTH;
    int num_tiles = (out_size + tile_width - 1) / tile_width;   // number of tiles needed to cover out_size

    // Compute maximum input tile length for allocation in shared memory
    int input_tile_len_max = (tile_width - 1) * stride + 1 + (kernel_size - 1) * dilation;

    // Grid dimensions: x: num_tiles (for spatial tiling), y: out_channels, z: B
    dim3 grid(num_tiles, out_channels, B);
    // Block dimension: use tile_width threads per block
    dim3 block(tile_width);

    // Dynamic shared memory size: space for input tile and weight filter
    size_t shared_mem_size = (in_channels * input_tile_len_max + in_channels * kernel_size) * sizeof(float);

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);
    float* output_data = output.data_ptr<float>();

    conv1d_kernel_shared<<<grid, block, shared_mem_size>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation,
        input_tile_len_max
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) leveraging shared memory for input and weight");
}
