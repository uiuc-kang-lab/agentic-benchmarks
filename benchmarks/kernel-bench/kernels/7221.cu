#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void conv2d_shared_kernel_streamed(
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding,
    int batch_offset) {

    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels + batch_offset;

    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sh_weight[];
    int filter_elems = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Asynchronously load kernel weights into shared memory using cp.async (requires Volta or newer architecture)
    int filter_offset = oc * filter_elems;
    for (int idx = tid; idx < filter_elems; idx += blockDim.x * blockDim.y) {
        // cp.async intrinsic to overlap global load with computation
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n"
            :
            : "r"(&sh_weight[idx]), "l"(&weight[filter_offset + idx]), "n"(sizeof(float))
        );
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    if (out_row >= out_h || out_col >= out_w) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll
        for (int ki = 0; ki < kernel_size; ki++) {
            #pragma unroll
            for (int kj = 0; kj < kernel_size; kj++) {
                int in_row = out_row * stride - padding + ki;
                int in_col = out_col * stride - padding + kj;
                if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                    int input_index = n * (in_channels * in_h * in_w) + 
                                    ic * (in_h * in_w) + 
                                    in_row * in_w + in_col;
                    int filter_index = ic * (kernel_size * kernel_size) + 
                                     ki * kernel_size + kj;
                    sum += input[input_index] * sh_weight[filter_index];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_index = n * (out_channels * out_h * out_w) + 
                    oc * (out_h * out_w) + 
                    out_row * out_w + out_col;
    output[out_index] = sum;
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

    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), 
                               {stride, stride}, {padding, padding}, 
                               {dilation, dilation}, groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), 
                               {stride, stride}, {padding, padding}, 
                               {dilation, dilation}, groups);
        }
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate batch chunks for each stream
    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    for (int i = 0; i < NUM_STREAMS; i++) {
        int stream_batch_offset = i * batches_per_stream;
        int stream_batch_size = std::min(batches_per_stream, 
                                       batch_size - stream_batch_offset);
        
        if (stream_batch_size <= 0) continue;

        dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
                 (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
                 stream_batch_size * out_channels);

        conv2d_shared_kernel_streamed<<<grid, block, shared_mem_size, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_size,
            out_h,
            out_w,
            stride,
            padding,
            stream_batch_offset
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with streams");
}