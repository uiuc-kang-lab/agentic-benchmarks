#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum allowed number of weight elements for constant memory
#define MAX_WEIGHT_SIZE 4096

// Constant memory for convolution weights
__constant__ float c_weight[MAX_WEIGHT_SIZE];

// Kernel implementing transposed 1D convolution with selective shared memory usage
// When stride == 1, a shared memory tile is loaded for input data to achieve coalesced and cached reads.
// __syncthreads() is called only once after loading shared memory.

// Grid dimensions:
//   blockIdx.x : batch index
//   blockIdx.y : output channel index
//   blockIdx.z : tile index along the output width dimension
// Block dimension:
//   blockDim.x : tile width (number of output elements processed in one block along spatial dimension)

// If stride != 1, the kernel falls back to direct global memory reads without shared memory.

extern "C"
__global__ void synced_shmem_conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int groups) {

    // Determine indices from grid and block
    int b = blockIdx.x; // batch index
    int o = blockIdx.y; // output channel index
    int tile_start = blockIdx.z * blockDim.x; // starting output index for this tile
    int j = tile_start + threadIdx.x; // output spatial index
    if (j >= output_width) return;

    // Determine group parameters: each group handles a subset of channels
    int group_size_out = out_channels / groups;  // output channels per group
    int group_in_channels = in_channels / groups;  // input channels per group
    int g = o / group_size_out;                   // group index
    int o_in_group = o % group_size_out;            // local output channel index

    float sum = 0.0f;

    // If stride is 1, we can load a contiguous tile of input data into shared memory
    if (stride == 1) {
        // The required input indices for an output j: i = j + padding - k for k in [0, kernel_size)
        // For a tile of output indices [tile_start, tile_start + blockDim.x - 1], the union of needed input indices is:
        // [tile_start + padding - (kernel_size - 1), tile_start + padding + blockDim.x - 1]
        int shmem_width = blockDim.x + kernel_size - 1;
        int start_load = tile_start + padding - (kernel_size - 1);

        // Declare extern shared memory
        extern __shared__ float shmem[]; // size: group_in_channels * shmem_width floats

        // Each thread loads multiple elements into shared memory for each input channel in the group
        int total_load = group_in_channels * shmem_width;
        for (int idx = threadIdx.x; idx < total_load; idx += blockDim.x) {
            int local_ch = idx / shmem_width; // channel index within the group [0, group_in_channels)
            int offset = idx % shmem_width;
            int global_i = start_load + offset; // desired input spatial index
            int input_idx = b * (in_channels * input_width) + (g * group_in_channels + local_ch) * input_width + global_i;
            if (global_i < 0 || global_i >= input_width) {
                shmem[idx] = 0.0f;
            } else {
                shmem[idx] = input[input_idx];
            }
        }
        __syncthreads(); // Ensure shared memory is fully loaded

        // Compute output for this thread
        // For each kernel element k, the corresponding input index is i = j + padding - k
        for (int k = 0; k < kernel_size; ++k) {
            int i_val = j + padding - k;  // since stride==1, no modulus condition
            int rel_index = i_val - start_load; // index in shared memory
            if (rel_index < 0 || rel_index >= shmem_width) continue;

            // Accumulate over all input channels in the group
            for (int ic = 0; ic < group_in_channels; ++ic) {
                int weight_index = ((g * group_in_channels + ic) * group_size_out + o_in_group) * kernel_size + k;
                float in_val = shmem[ic * shmem_width + rel_index];
                sum += in_val * c_weight[weight_index];
            }
        }
    } else {
        // When stride != 1, perform direct global memory accesses
        for (int k = 0; k < kernel_size; ++k) {
            int i_val = j + padding - k;
            if (i_val % stride != 0) continue;
            int i_idx = i_val / stride;
            if (i_idx < 0 || i_idx >= input_width) continue;
            for (int ic = 0; ic < group_in_channels; ++ic) {
                int input_idx = b * (in_channels * input_width) + (g * group_in_channels + ic) * input_width + i_idx;
                int weight_index = ((g * group_in_channels + ic) * group_size_out + o_in_group) * kernel_size + k;
                sum += input[input_idx] * c_weight[weight_index];
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[o];
    }

    // Write the computed sum to the output tensor
    int out_idx = b * (out_channels * output_width) + o * output_width + j;
    output[out_idx] = sum;
}

// Host wrapper function
torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_width = x.size(2);
    int kernel_size = weight.size(2);
    int group_size_out = weight.size(1); // assuming weight shape: [in_channels, group_size_out, kernel_size]
    int out_channels = group_size_out * groups;
    
    // Calculate output width based on transposed convolution formula
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());

    // Check that weight size does not exceed constant memory capacity
    int num_weight_elems = weight.numel();
    TORCH_CHECK(num_weight_elems <= MAX_WEIGHT_SIZE, "Weight size exceeds constant memory limit");
    cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), num_weight_elems * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Set up kernel launch configuration
    // Use a tile width for the output spatial dimension
    int tile_width = 256;
    dim3 block(tile_width);
    // Grid: one block per batch and output channel, and tile over output width
    dim3 grid(batch_size, out_channels, (output_width + tile_width - 1) / tile_width);

    // Determine shared memory size needed when stride==1
    int group_in_channels = in_channels / groups;  
    size_t shmem_size = 0;
    if (stride == 1) {
        // Shared memory size for one block: group_in_channels * (tile_width + kernel_size - 1) floats
        shmem_size = group_in_channels * (tile_width + kernel_size - 1) * sizeof(float);
    }

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Launch the kernel on the current CUDA stream
    synced_shmem_conv1d_kernel<<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA) with shared memory and minimal synchronization");
}
