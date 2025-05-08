#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 16

__global__ void depthwise_conv2d_shared_memory(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[TILE_SIZE * TILE_SIZE];

    const int APRON = ((kernel_h - 1) * dilation) / 2;
    
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;
    
    int tile_start_h = blockIdx.y * TILE_SIZE;
    int tile_start_w = blockIdx.x * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load weights into shared memory
    if (ty == 0 && tx < kernel_h) {
        shared_weight[tx] = __ldg(&weight[c * kernel_h + tx]);
    }
    
    // Calculate input tile dimensions including apron
    int tile_h = min(TILE_SIZE + 2 * APRON, in_h - tile_start_h + 2 * APRON);
    int tile_w = min(TILE_SIZE + 2 * APRON, in_w - tile_start_w + 2 * APRON);

    // Load input tile into shared memory
    for (int i = ty; i < tile_h; i += TILE_SIZE) {
        for (int j = tx; j < tile_w; j += TILE_SIZE) {
            int ih = tile_start_h + i - APRON;
            int iw = tile_start_w + j - APRON;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                shared_input[i * (TILE_SIZE + 2 * APRON) + j] = 
                    __ldg(&input[((b * channels + c) * in_h + ih) * in_w + iw]);
            } else {
                shared_input[i * (TILE_SIZE + 2 * APRON) + j] = 0.0f;
            }
        }
    }
    
    __syncthreads();

    int oh = tile_start_h + ty;
    int ow = tile_start_w + tx;

    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih = ty + kh * dilation;
            int iw = tx;
            sum += shared_input[ih * (TILE_SIZE + 2 * APRON) + iw] * shared_weight[kh];
        }
        
        sum += __ldg(&bias[c]);
        output[((b * channels + c) * out_h + oh) * out_w + ow] = sum;
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == channels");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    const int APRON = ((kernel_h - 1) * dilation) / 2;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * channels);

    int shared_mem_size = ((TILE_SIZE + 2 * APRON) * (TILE_SIZE + 2 * APRON) + kernel_h) * sizeof(float);

    depthwise_conv2d_shared_memory<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with shared memory tiling (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}