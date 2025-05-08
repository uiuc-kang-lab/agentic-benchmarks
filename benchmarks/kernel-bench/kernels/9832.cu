#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define NUM_STREAMS 4

__global__ void depthwise_conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int output_h,
    int output_w,
    int kernel_size,
    int stride,
    int padding,
    int channels_per_group,
    int batch_offset
) {
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_weight = shared_mem + ((TILE_HEIGHT-1)*stride + kernel_size) * ((TILE_WIDTH-1)*stride + kernel_size);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int b = bz / out_channels + batch_offset;
    int oc = bz % out_channels;
    int in_ch = oc / channels_per_group;
    int weight_ch = oc % channels_per_group;

    // Pre-compute input/output positions
    int out_x = bx * TILE_WIDTH + tx;
    int out_y = by * TILE_HEIGHT + ty;
    
    // Load weights into shared memory cooperatively
    int thread_id = ty * blockDim.x + tx;
    int total_weight_elements = kernel_size * kernel_size;
    for(int idx = thread_id; idx < total_weight_elements; idx += blockDim.x * blockDim.y) {
        int weight_offset = in_ch * (channels_per_group * kernel_size * kernel_size) +
                           weight_ch * (kernel_size * kernel_size);
        s_weight[idx] = weight[weight_offset + idx];
    }

    // Calculate input tile dimensions
    int in_tile_width = (TILE_WIDTH-1)*stride + kernel_size;
    int in_tile_height = (TILE_HEIGHT-1)*stride + kernel_size;
    
    // Load input tile into shared memory
    int in_tile_start_x = bx * TILE_WIDTH * stride - padding;
    int in_tile_start_y = by * TILE_HEIGHT * stride - padding;
    
    for(int i = thread_id; i < in_tile_height * in_tile_width; i += blockDim.x * blockDim.y) {
        int tile_y = i / in_tile_width;
        int tile_x = i % in_tile_width;
        int in_y = in_tile_start_y + tile_y;
        int in_x = in_tile_start_x + tile_x;
        
        float val = 0.0f;
        if(in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
            val = input[b * (in_channels * input_h * input_w) +
                       in_ch * (input_h * input_w) +
                       in_y * input_w + in_x];
        }
        s_input[tile_y * in_tile_width + tile_x] = val;
    }
    
    __syncthreads();

    if(out_x < output_w && out_y < output_h) {
        float sum = 0.0f;
        #pragma unroll
        for(int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for(int kx = 0; kx < kernel_size; kx++) {
                int in_y = ty * stride + ky;
                int in_x = tx * stride + kx;
                sum += s_input[in_y * in_tile_width + in_x] *
                       s_weight[ky * kernel_size + kx];
            }
        }
        
        if(bias != nullptr) {
            sum += bias[oc];
        }
        
        output[b * (out_channels * output_h * output_w) +
               oc * (output_h * output_w) +
               out_y * output_w +
               out_x] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int channels_per_group = weight.size(1);
    const int out_channels = in_channels * channels_per_group;
    
    const int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    const int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate shared memory size
    const int smem_size = (((TILE_HEIGHT-1)*stride + kernel_size) * 
                          ((TILE_WIDTH-1)*stride + kernel_size) + 
                          kernel_size * kernel_size) * sizeof(float);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 numBlocks(
        (output_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        out_channels
    );

    // Process batch in chunks using different streams
    const int batch_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for(int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
        const int batch_start = stream_idx * batch_per_stream;
        const int batch_end = std::min(batch_start + batch_per_stream, batch_size);
        if(batch_start >= batch_size) continue;

        const int current_batch_size = batch_end - batch_start;
        
        // Calculate input/output offsets for this batch chunk
        const float* input_ptr = input.data_ptr<float>() + 
                               batch_start * in_channels * input_h * input_w;
        float* output_ptr = output.data_ptr<float>() + 
                           batch_start * out_channels * output_h * output_w;

        depthwise_conv2d_shared_kernel<<<numBlocks, threadsPerBlock, smem_size, streams[stream_idx]>>>(
            input_ptr,
            weight.data_ptr<float>(),
            bias.has_value() ? bias->data_ptr<float>() : nullptr,
            output_ptr,
            current_batch_size,
            in_channels,
            input_h,
            input_w,
            out_channels,
            output_h,
            output_w,
            kernel_size,
            stride,
            padding,
            channels_per_group,
            batch_start
        );
    }

    // Synchronize and cleanup streams
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution with Pipelined Memory Operations",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"));
}