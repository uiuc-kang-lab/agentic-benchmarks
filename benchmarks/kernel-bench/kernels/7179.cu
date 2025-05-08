#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int NUM_STREAMS = 4;
constexpr int TILE_SIZE = 16;

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_start,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    __shared__ float shared_input[TILE_SIZE * TILE_SIZE];
    __shared__ float shared_weight[TILE_SIZE * TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z + batch_start;

    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;

    if (out_x >= out_width || out_y >= out_height) return;

    float sum = 0.0f;

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kernel_size; ky += TILE_SIZE) {
                for (int kx = 0; kx < kernel_size; kx += TILE_SIZE) {
                    if (tx < TILE_SIZE && ty < TILE_SIZE) {
                        const int in_y = out_y * stride - padding + (ky + ty) * dilation;
                        const int in_x = out_x * stride - padding + (kx + tx) * dilation;
                        
                        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                            shared_input[ty * TILE_SIZE + tx] = 
                                input[((bz * in_channels + ic) * in_height + in_y) * in_width + in_x];
                        } else {
                            shared_input[ty * TILE_SIZE + tx] = 0.0f;
                        }
                        
                        if ((ky + ty) < kernel_size && (kx + tx) < kernel_size) {
                            shared_weight[ty * TILE_SIZE + tx] = 
                                weight[((oc * in_channels + ic) * kernel_size + ky + ty) * kernel_size + kx + tx];
                        }
                    }
                    __syncthreads();

                    #pragma unroll
                    for (int i = 0; i < TILE_SIZE; ++i) {
                        if ((ky + i) < kernel_size && (kx + tx) < kernel_size) {
                            sum += shared_input[ty * TILE_SIZE + i] * shared_weight[i * TILE_SIZE + tx];
                        }
                    }
                    __syncthreads();
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[oc];
        }

        if (out_x < out_width && out_y < out_height) {
            output[((bz * out_channels + oc) * out_height + out_y) * out_width + out_x] = sum;
        }
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

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch, out_channels, out_height, out_width}, x.options());

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int batch_per_stream = (batch + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_width + TILE_SIZE - 1) / TILE_SIZE,
              (out_height + TILE_SIZE - 1) / TILE_SIZE,
              batch_per_stream);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int batch_start = i * batch_per_stream;
        if (batch_start >= batch) break;

        conv2d_kernel<<<grid, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_start,
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
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed CUDA convolution implementation");
}
