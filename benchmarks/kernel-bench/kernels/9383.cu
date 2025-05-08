#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for output (each block computes a TILE_OUT_H x TILE_OUT_W tile of the output feature map)
#define TILE_OUT_W 32
#define TILE_OUT_H 8  // 32 * 8 = 256 threads per block, good for occupancy

// CUDA kernel using shared memory tiling for input reuse
__global__ void conv2d_shared_mem_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    // Grid mapping: each block processes a tile for one output channel of one batch sample
    // blockIdx.z: combined index for batch and output channel
    int b_oc = blockIdx.z;
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;

    // Determine the top-left coordinates of this block's output tile
    int tile_out_x = blockIdx.x * TILE_OUT_W;
    int tile_out_y = blockIdx.y * TILE_OUT_H;

    // Compute corresponding top-left corner in the input (for the tile's receptive field)
    // Each output pixel is computed from input starting at (out_index * stride - pad)
    int sm_origin_x = tile_out_x * stride - pad_w;
    int sm_origin_y = tile_out_y * stride - pad_h;

    // Compute the dimensions of the shared memory tile (the input patch needed by the whole output tile)
    // For an output tile of size TILE_OUT_W x TILE_OUT_H, the input patch size is:
    //   sm_w = (TILE_OUT_W - 1)*stride + (kernel_w - 1)*dilation_w + 1
    //   sm_h = (TILE_OUT_H - 1)*stride + (kernel_h - 1)*dilation_h + 1
    int sm_w = (TILE_OUT_W - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    int sm_h = (TILE_OUT_H - 1) * stride + (kernel_h - 1) * dilation_h + 1;

    // Declare external shared memory. This will store one channel's input patch per iteration.
    extern __shared__ float sdata[]; // Size: sm_w * sm_h floats

    // Initialize accumulator with bias if provided
    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    // Loop over each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // Load the required input patch for the current input channel into shared memory
        // Use all threads in the block to cooperatively load sm_w * sm_h elements
        int num_elements = sm_w * sm_h;
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = threadId; idx < num_elements; idx += blockDim.x * blockDim.y) {
            int local_y = idx / sm_w;
            int local_x = idx % sm_w;
            int global_y = sm_origin_y + local_y;
            int global_x = sm_origin_x + local_x;

            if (global_y >= 0 && global_y < input_height && global_x >= 0 && global_x < input_width) {
                sdata[idx] = x[b * (in_channels * input_height * input_width) +
                               ic * (input_height * input_width) +
                               global_y * input_width + global_x];
            } else {
                sdata[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Each thread computes its corresponding output element's contribution for this input channel
        int out_x = tile_out_x + threadIdx.x;
        int out_y = tile_out_y + threadIdx.y;
        if (out_x < out_width && out_y < out_height) {
            float partial = 0.0f;
            // For this output element, the top-left of its receptive field in shared memory is at (threadIdx.y*stride, threadIdx.x*stride)
            int local_base_y = threadIdx.y * stride;
            int local_base_x = threadIdx.x * stride;
            
            // Loop over the kernel
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int sm_y = local_base_y + kh * dilation_h;
                    int sm_x = local_base_x + kw * dilation_w;
                    int sm_index = sm_y * sm_w + sm_x;
                    float input_val = sdata[sm_index];
                    // Weight is stored with layout: [out_channels, in_channels, kernel_h, kernel_w]
                    int weight_index = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    partial += input_val * weight[weight_index];
                }
            }
            sum += partial;
        }
        __syncthreads(); // Ensure all threads have finished using sdata before the next ic iteration
    }

    // Write the output for this thread if it is within valid output bounds
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;
    if (out_x < out_width && out_y < out_height) {
        int out_index = b * (out_channels * out_height * out_width) +
                        oc * (out_height * out_width) +
                        out_y * out_width + out_x;
        output[out_index] = sum;
    }
}

// Forward function wrapping the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // optional bias
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    // Calculate output dimensions using convolution formula
    int out_height = (input_height + 2 * pad_h - (kernel_h - 1) * dilation_h - 1) / stride + 1;
    int out_width = (input_width + 2 * pad_w - (kernel_w - 1) * dilation_w - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Setup grid and block dimensions
    dim3 block(TILE_OUT_W, TILE_OUT_H);
    int grid_x = (out_width + TILE_OUT_W - 1) / TILE_OUT_W;
    int grid_y = (out_height + TILE_OUT_H - 1) / TILE_OUT_H;
    int grid_z = batch_size * out_channels;  // each block in z corresponds to one (batch, out_channel)
    dim3 grid(grid_x, grid_y, grid_z);

    // Compute shared memory size per block
    int sm_w = (TILE_OUT_W - 1) * stride + (kernel_w - 1) * dilation_w + 1;
    int sm_h = (TILE_OUT_H - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    size_t shared_mem_size = sm_w * sm_h * sizeof(float);

    // Launch the kernel
    conv2d_shared_mem_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        kernel_h,
        kernel_w,
        out_height,
        out_width,
        stride,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CUDA) with shared memory");
}
