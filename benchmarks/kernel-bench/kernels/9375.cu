#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel is similar to our previous implementation but includes a batch_offset
// parameter so that it can operate on a sub-batch. Using multiple CUDA streams at the host level
// we overlap any asynchronous memory transfers (if present) with kernel execution, thus pipelining
// computation and memory operations.

#define CHANNELS_PER_BLOCK 4
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Kernel: Conv2D with pipelined execution across sub-batches (using batch_offset parameter)
__global__ void conv2d_kernel_pipelined(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_batch,           // overall total batch size
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int batch_offset) {      // offset of the sub-batch

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;  // Encodes both the sub-batch index and output channel group

    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;

    // Compute output spatial coordinates
    int h_out = by * blockDimY + ty;
    int w_out = bx * blockDimX + tx;

    // Determine the number of output channel groups
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int local_batch = bz / groups_per_batch;      // local index within this sub-batch
    int g = bz % groups_per_batch;                // output channel group index
    int b = batch_offset + local_batch;           // global batch index

    // Return if thread is outside the needed range
    if (b >= total_batch || h_out >= height_out || w_out >= width_out)
        return;

    int oc_start = g * CHANNELS_PER_BLOCK;

    // Initialize accumulators; load bias if provided
    float sums[CHANNELS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            sums[i] = (bias != nullptr) ? bias[global_oc] : 0.0f;
        }
    }

    // Cooperative loading of the weight tile for this output channel group into shared memory
    // Shared memory size: CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w
    extern __shared__ float shared_weight[];
    int total_weight_elems = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    int threadId = ty * blockDimX + tx;
    int blockSize = blockDimX * blockDimY;
    for (int idx = threadId; idx < total_weight_elems; idx += blockSize) {
        int w_oc = idx / (in_channels * kernel_h * kernel_w);
        int rem = idx % (in_channels * kernel_h * kernel_w);
        int global_oc = oc_start + w_oc;
        shared_weight[idx] = (global_oc < out_channels) ? weight[global_oc * in_channels * kernel_h * kernel_w + rem] : 0.0f;
    }
    __syncthreads();

    // Offsets for the current batch
    int x_batch_offset = b * in_channels * input_height * input_width;
    int out_batch_offset = b * out_channels * height_out * width_out;

    // Compute convolution: loop over input channels and kernel spatial dimensions
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int h_in = h_out * stride + kh * dilation_h - pad_h;
            bool valid_h = (h_in >= 0 && h_in < input_height);
            for (int kw = 0; kw < kernel_w; kw++) {
                int w_in = w_out * stride + kw * dilation_w - pad_w;
                bool valid_w = (w_in >= 0 && w_in < input_width);
                float x_val = (valid_h && valid_w) ? __ldg(&x[x_batch_offset + ic * input_height * input_width + h_in * input_width + w_in]) : 0.0f;
                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                    int weight_offset = i * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw;
                    sums[i] += x_val * shared_weight[weight_offset];
                }
            }
        }
    }

    // Write computed output values to global memory
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            int out_idx = out_batch_offset + global_oc * height_out * width_out + h_out * width_out + w_out;
            output[out_idx] = sums[i];
        }
    }
}

// Forward function: Launches the kernel over multiple CUDA streams to overlap computation
// with potential memory transfer latency. We partition the batch dimension across streams.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
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

    int total_batch = x.size(0);
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

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({total_batch, out_channels, height_out, width_out}, x.options());

    // Use multiple CUDA streams to overlap kernel execution with memory transfers if any
    int num_streams = (total_batch < 4) ? total_batch : 4; // Use up to 4 streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
    }

    // Block and grid dimensions
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    size_t shared_mem_size = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w * sizeof(float);

    // Partition the batch across streams
    int base = total_batch / num_streams;
    int rem = total_batch % num_streams;
    int batch_start = 0;

    for (int s = 0; s < num_streams; s++) {
        int current_batch = base + (s < rem ? 1 : 0);
        if (current_batch == 0) continue;

        dim3 blocks(
            (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
            (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
            current_batch * groups_per_batch
        );

        conv2d_kernel_pipelined<<<blocks, threads, shared_mem_size, streams[s]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            total_batch,
            in_channels,
            input_height,
            input_width,
            out_channels,
            kernel_h,
            kernel_w,
            height_out,
            width_out,
            stride,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            batch_start  // batch_offset
        );

        batch_start += current_batch;
    }

    // Synchronize and destroy streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined Conv2D forward (CUDA)");
}
