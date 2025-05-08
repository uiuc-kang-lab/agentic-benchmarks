#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel minimizes warp divergence by replacing conditional branches in boundary checks
// with branchless arithmetic (using clamping and valid masks). Shared memory tiling is used
// to load the input patch for each thread block uniformly. The inner convolution loops are
// unrolled to further aid the compiler in producing uniform control flow within each warp.

__global__ void conv2d_nodivergence_kernel(
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

    // Determine output pixel coordinates and corresponding batch and output channel
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int n_oc = blockIdx.z; // combined index for batch and output channel
    int n = n_oc / out_channels;
    int oc = n_oc % out_channels;

    // Define shared memory tile dimensions (for the input patch corresponding to this block)
    int tile_sh_w = blockDim.x * stride + (kernel_size - 1) * dilation;
    int tile_sh_h = blockDim.y * stride + (kernel_size - 1) * dilation;
    int sh_tile_size = tile_sh_w * tile_sh_h;
    extern __shared__ float shared_tile[]; // dynamically allocated shared memory

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < in_channels; ic++) {
        // Compute pointer offset for this input channel
        int input_base = n * in_channels * in_height * in_width + ic * in_height * in_width;

        // Compute the top-left corner of the input patch (shared tile) for this block
        int in_tile_x = (blockIdx.x * blockDim.x) * stride - padding;
        int in_tile_y = (blockIdx.y * blockDim.y) * stride - padding;

        // Cooperative load of the shared memory tile in a branchless manner
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int num_threads = blockDim.x * blockDim.y;
        for (int i = tid; i < sh_tile_size; i += num_threads) {
            int sh_y = i / tile_sh_w;
            int sh_x = i % tile_sh_w;
            int global_x = in_tile_x + sh_x;
            int global_y = in_tile_y + sh_y;
            
            // Compute valid mask: 1 if the coordinate is in bounds, 0 otherwise
            int valid = ((global_x >= 0) && (global_x < in_width) && (global_y >= 0) && (global_y < in_height));
            
            // Compute clamped coordinates using branchless min/max operations
            int x_clamped = max(0, min(global_x, in_width - 1));
            int y_clamped = max(0, min(global_y, in_height - 1));
            
            // Load input value and multiply by valid flag so that out-of-bound accesses contribute 0
            float in_val = input[input_base + y_clamped * in_width + x_clamped];
            shared_tile[i] = valid * in_val;
        }
        __syncthreads();

        // Only threads corresponding to valid output positions perform convolution
        if (out_x < out_width && out_y < out_height) {
            // Compute starting offset in shared memory for the convolution window
            int sh_offset_x = threadIdx.x * stride;
            int sh_offset_y = threadIdx.y * stride;
            
            #pragma unroll
            for (int ky = 0; ky < kernel_size; ky++) {
                #pragma unroll
                for (int kx = 0; kx < kernel_size; kx++) {
                    int sh_x = sh_offset_x + kx * dilation;
                    int sh_y = sh_offset_y + ky * dilation;
                    float tile_val = shared_tile[sh_y * tile_sh_w + sh_x];
                    // Weight layout is [out_channels, in_channels, kernel_size, kernel_size]
                    float w = weight[ ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx ];
                    sum += tile_val * w;
                }
            }
        }
        __syncthreads(); // Ensure shared memory is reused safely for the next input channel
    }
    
    // Write the computed output if within bounds
    if (out_x < out_width && out_y < out_height) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_index = n * (out_channels * out_height * out_width) + oc * (out_height * out_width) + out_y * out_width + out_x;
        output[out_index] = sum;
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
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    // This custom kernel supports only groups==1 (the most common case)
    TORCH_CHECK(groups == 1, "conv2d_nodivergence_kernel supports only groups==1");

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width  = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);  // assuming square kernel

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    // Block dimensions: 16x16 output pixels per block
    dim3 block(16, 16);
    // Grid dimensions: cover entire output, with combined batch and output channel in z
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch * out_channels);

    // Calculate shared memory size in bytes
    int tile_sh_w = block.x * stride + (kernel_size - 1) * dilation;
    int tile_sh_h = block.y * stride + (kernel_size - 1) * dilation;
    size_t shared_mem_size = tile_sh_w * tile_sh_h * sizeof(float);

    conv2d_nodivergence_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Convolution with minimized warp divergence via branchless logic");
}
