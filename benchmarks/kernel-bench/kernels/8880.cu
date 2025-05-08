#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// This kernel leverages shared memory to cache group-specific weights, reducing global memory accesses.
// Threads within each block cooperatively load the weights into shared memory using a single __syncthreads(),
// and then compute a tile of the output for a given batch and group. This minimizes synchronization overhead
// by synchronizing only once after loading shared memory.

// Kernel: gridDim.x -> batch index (n), gridDim.y -> group index (g), gridDim.z -> tile index (across output channels, height, and width)
// Block dimensions: blockDim.x -> tile size for output channels (within group),
//                   blockDim.y -> tile size for output height,
//                   blockDim.z -> tile size for output width.

__global__ void conv_transpose2d_kernel_shared(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    // Determine the batch index and group index from grid dimensions
    int n = blockIdx.x; // batch index
    int g = blockIdx.y; // group index

    // Calculate the number of weight elements for this group
    const int weight_group_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w;

    // Allocate shared memory for the weight of the current group
    extern __shared__ float shared_weight[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (int i = tid; i < weight_group_size; i += blockSize) {
        shared_weight[i] = weight[g * weight_group_size + i];
    }
    // Synchronize to ensure all weight data is loaded into shared memory
    __syncthreads();

    // Define the tile dimensions from the block dimensions
    int tile_oc = blockDim.x; // tile size for output channels within the group
    int tile_oh = blockDim.y; // tile size for output height
    int tile_ow = blockDim.z; // tile size for output width

    // Compute the number of tiles in each dimension for this group
    int n_tiles_oc = (out_channels_per_group + tile_oc - 1) / tile_oc;
    int n_tiles_oh = (out_h + tile_oh - 1) / tile_oh;
    int n_tiles_ow = (out_w + tile_ow - 1) / tile_ow;

    // Decode blockIdx.z to obtain the tile indices
    int tile_index = blockIdx.z;
    int tile_oc_index = tile_index % n_tiles_oc;
    int temp = tile_index / n_tiles_oc;
    int tile_oh_index = temp % n_tiles_oh;
    int tile_ow_index = temp / n_tiles_oh;

    // Compute starting indices for this tile
    int oc_start = tile_oc_index * tile_oc;  // output channel offset within the group
    int oh_start = tile_oh_index * tile_oh;
    int ow_start = tile_ow_index * tile_ow;

    // Each thread in the block corresponds to one element in the tile
    int local_oc = threadIdx.x; // local output channel index within the tile
    int local_oh = threadIdx.y; // local output height index within the tile
    int local_ow = threadIdx.z; // local output width index within the tile

    // Compute global indices for the output element
    int oc_local = oc_start + local_oc;            // channel index within the group
    int global_oc = g * out_channels_per_group + oc_local; // global output channel index
    int global_oh = oh_start + local_oh;
    int global_ow = ow_start + local_ow;

    // Return if the thread is out of the valid range
    if (oc_local >= out_channels_per_group || global_oh >= out_h || global_ow >= out_w) {
        return;
    }

    // Initialize the output value with the corresponding bias
    float out_val = bias[global_oc];

    // Compute candidate positions (output coordinate plus padding)
    int candidate_h = global_oh + pad_h;
    int candidate_w = global_ow + pad_w;

    // Loop over the kernel window
    for (int kh = 0; kh < kernel_h; kh++) {
        int h_in_candidate = candidate_h - kh * dilation_h;
        if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
        int ih = h_in_candidate / stride_h;
        if (ih >= in_h) continue;

        for (int kw = 0; kw < kernel_w; kw++) {
            int w_in_candidate = candidate_w - kw * dilation_w;
            if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
            int iw = w_in_candidate / stride_w;
            if (iw >= in_w) continue;

            // Accumulate over input channels for this group
            for (int c = 0; c < in_channels_per_group; c++) {
                int x_index = n * (in_channels * in_h * in_w) +
                              (g * in_channels_per_group + c) * (in_h * in_w) +
                              ih * in_w + iw;
                // Weight layout in shared memory: [in_channels_per_group, out_channels_per_group, kernel_h, kernel_w]
                int weight_index = c * (out_channels_per_group * kernel_h * kernel_w) +
                                   (oc_local) * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw;
                out_val += x[x_index] * shared_weight[weight_index];
            }
        }
    }

    // Write the computed output value back to global memory
    int out_index = n * (out_channels * out_h * out_w) +
                    global_oc * (out_h * out_w) +
                    global_oh * out_w + global_ow;
    output[out_index] = out_val;
}

// Host wrapper function
at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {

    // Ensure input tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.has_value() && bias.value().defined())
        bias = bias.value().contiguous();

    // Get dimensions
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1); // weight layout: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    const int out_channels = out_channels_per_group * groups;

    // Convolution parameters
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    // Compute output dimensions for conv_transpose2d
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    // If bias is not provided, create a zero tensor
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({out_channels}, weight.options());
    }

    auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());

    int in_channels_per_group = in_channels / groups;

    // Define block (tile) dimensions
    // Here we choose a tile of 8 channels, 8 height, and 1 width per block
    dim3 blockDim(8, 8, 1);

    // Calculate number of tiles in each dimension within each group
    int n_tiles_oc = (out_channels_per_group + blockDim.x - 1) / blockDim.x;
    int n_tiles_oh = (out_h + blockDim.y - 1) / blockDim.y;
    int n_tiles_ow = (out_w + blockDim.z - 1) / blockDim.z;
    int grid_tiles = n_tiles_oc * n_tiles_oh * n_tiles_ow;

    // Set grid dimensions:
    //   gridDim.x: batch
    //   gridDim.y: groups
    //   gridDim.z: tile index covering output channels (within group), height, and width
    dim3 gridDim(batch, groups, grid_tiles);

    // Allocate shared memory size for one group's weights (in bytes)
    size_t shared_mem_size = in_channels_per_group * out_channels_per_group * kernel_h * kernel_w * sizeof(float);

    conv_transpose2d_kernel_shared<<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels_per_group,
        out_channels_per_group
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Transposed Convolution with Shared Weight and Minimal Syncthreads (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
