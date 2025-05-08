#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Tile dimensions for output; these can be tuned for performance
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Kernel: Shared-memory based depthwise convolution for a sub-batch
// This kernel computes convolution for 'local_batch' samples.

__global__ void depthwise_conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int local_batch, // number of samples in this sub-batch
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group
) {
    // Each block handles a tile for one (batch, output channel) pair
    // blockIdx.z spans [0, local_batch * out_channels)
    int bat_oc = blockIdx.z;
    int b = bat_oc / out_channels;  // local index within the sub-batch
    int oc = bat_oc % out_channels;

    int in_ch = oc / channels_per_group;
    // weight channel index: (oc % channels_per_group)

    // Determine the starting coordinates of the output tile
    int tile_out_x = blockIdx.x * TILE_WIDTH;
    int tile_out_y = blockIdx.y * TILE_HEIGHT;

    // Each thread computes one output pixel within the tile
    int out_x = tile_out_x + threadIdx.x;
    int out_y = tile_out_y + threadIdx.y;

    // Calculate the top-left input coordinate corresponding to this tile
    int in_start_y = tile_out_y * stride - padding;
    int in_start_x = tile_out_x * stride - padding;

    // Dimensions of shared memory needed for the input tile
    int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
    int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;

    extern __shared__ float shared_mem[];
    // First portion for caching the input tile, second for the kernel
    float* s_input = shared_mem;
    float* s_weight = shared_mem + smem_rows * smem_cols;

    // Linear thread index for cooperative loading
    int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;

    // Load kernel weights into shared memory
    int num_weight = kernel_size * kernel_size;
    for (int i = linear_thread; i < num_weight; i += blockDim.x * blockDim.y) {
        s_weight[i] = weight[
            in_ch * (channels_per_group * kernel_size * kernel_size) +
            (oc % channels_per_group) * (kernel_size * kernel_size) +
            i
        ];
    }

    // Load input tile into shared memory
    int total_input = smem_rows * smem_cols;
    for (int i = linear_thread; i < total_input; i += blockDim.x * blockDim.y) {
        int r = i / smem_cols;
        int c = i % smem_cols;
        int global_y = in_start_y + r;
        int global_x = in_start_x + c;
        float val = 0.0f;
        if (global_y >= 0 && global_y < input_h && global_x >= 0 && global_x < input_w) {
            int input_idx = b * (in_channels * input_h * input_w) +
                            in_ch * (input_h * input_w) +
                            global_y * input_w + global_x;
            val = input[input_idx];
        }
        s_input[i] = val;
    }

    __syncthreads();

    // Compute convolution if the thread maps to a valid output pixel
    if (out_y < output_h && out_x < output_w) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int s_y = threadIdx.y * stride + ky;
                int s_x = threadIdx.x * stride + kx;
                float in_val = s_input[s_y * smem_cols + s_x];
                float wt = s_weight[ky * kernel_size + kx];
                sum += in_val * wt;
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = b * (out_channels * output_h * output_w) +
                      oc * (output_h * output_w) +
                      out_y * output_w + out_x;
        output[out_idx] = sum;
    }
}


// Forward function that pipelines kernel computation and memory transfers using 2 CUDA streams and double buffering
// The approach divides the batch into sub-batches. For each sub-batch, the kernel is launched asynchronously to
// compute the output into a temporary device buffer. Immediately after, an asynchronous copy transfers the
// computed results to a pinned host buffer. After all chunks are processed, the pinned host buffer is copied
// back to a final GPU tensor. This overlap hides memory transfer latency with computation.

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Input and weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }
    TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_size = weight.size(2);
    int channels_per_group = weight.size(1);
    int out_channels = in_channels * channels_per_group;
    const float* bias_ptr = (bias.has_value()) ? bias->data_ptr<float>() : nullptr;

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    int output_elements_per_sample = out_channels * output_h * output_w;
    size_t total_output_bytes = batch_size * output_elements_per_sample * sizeof(float);

    // Use 2 streams for pipelining (double buffering)
    int num_streams = 2;
    int chunk_size = (batch_size + num_streams - 1) / num_streams;
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    // Allocate pinned host memory to stage the complete output
    float* host_output = nullptr;
    cudaMallocHost((void**)&host_output, total_output_bytes);

    // Allocate temporary device buffers for each stream (max size for a chunk)
    std::vector<float*> d_temp(num_streams, nullptr);
    size_t temp_buffer_size = chunk_size * output_elements_per_sample * sizeof(float);
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc((void**)&d_temp[i], temp_buffer_size);
    }

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // For each chunk, launch the kernel asynchronously and then copy results from the temporary device
    // buffer to the pinned host memory concurrently.
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int current_batch = std::min(chunk_size, batch_size - chunk * chunk_size);

        // Launch kernel for this sub-batch
        int grid_x = (output_w + TILE_WIDTH - 1) / TILE_WIDTH;
        int grid_y = (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
        int grid_z = current_batch * out_channels;  // one block per (sample, channel)
        dim3 grid(grid_x, grid_y, grid_z);
        dim3 block(TILE_WIDTH, TILE_HEIGHT);

        int smem_rows = (TILE_HEIGHT - 1) * stride + kernel_size;
        int smem_cols = (TILE_WIDTH - 1) * stride + kernel_size;
        size_t shared_mem_bytes = (smem_rows * smem_cols + kernel_size * kernel_size) * sizeof(float);

        // Offset pointer for the input chunk
        const float* input_chunk = input.data_ptr<float>() + (chunk * chunk_size) * (in_channels * input_h * input_w);

        int stream_id = chunk % num_streams;
        float* d_temp_output = d_temp[stream_id];

        depthwise_conv2d_shared_kernel<<<grid, block, shared_mem_bytes, streams[stream_id]>>>(
            input_chunk,
            weight.data_ptr<float>(),
            bias_ptr,
            d_temp_output,
            current_batch,
            in_channels,
            input_h,
            input_w,
            out_channels,
            output_h,
            output_w,
            kernel_size,
            stride,
            padding,
            channels_per_group
        );

        size_t chunk_bytes = current_batch * output_elements_per_sample * sizeof(float);
        // Asynchronously copy the computed output from device to pinned host
        cudaMemcpyAsync(host_output + (chunk * chunk_size) * output_elements_per_sample,
                        d_temp_output,
                        chunk_bytes,
                        cudaMemcpyDeviceToHost,
                        streams[stream_id]);
    }

    // Synchronize all streams to ensure that kernels and memcpy operations complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Allocate the final output tensor on GPU
    auto final_output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    // Copy the complete output from pinned host memory back to the GPU
    cudaMemcpy(final_output.data_ptr<float>(), host_output, total_output_bytes, cudaMemcpyHostToDevice);

    // Cleanup: free pinned host memory, temporary device buffers, and destroy streams
    cudaFreeHost(host_output);
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_temp[i]);
        cudaStreamDestroy(streams[i]);
    }

    return final_output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with Pipelined Streams and Asynchronous Memory Transfers (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("stride"), py::arg("padding"));
}
