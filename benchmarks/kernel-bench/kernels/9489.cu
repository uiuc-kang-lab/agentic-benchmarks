#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Combined CUDA kernel: leverages shared memory for weight caching and __ldg for read-only input/bias accesses,
// along with loop tiling for input channels to enhance computation efficiency.
__global__ void conv_transpose2d_forward_kernel_shared_ldg(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {

    // Compute output pixel coordinates
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int bo_idx = blockIdx.z;
    int o = bo_idx % out_channels;
    int b = bo_idx / out_channels;

    if (out_w >= out_width || out_h >= out_height)
        return;

    // Allocate shared memory for the weight tile for this output channel
    extern __shared__ float shared_weight[]; // size: in_channels * kernel_size * kernel_size
    int weight_count = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread cooperatively loads a subset of weights from global memory to shared memory
    for (int i = tid; i < weight_count; i += blockDim.x * blockDim.y) {
        int tmp = i;
        int q = tmp % kernel_size;
        tmp /= kernel_size;
        int p = tmp % kernel_size;
        int c = tmp / kernel_size;
        // Compute global memory index for weight (shape: [in_channels, out_channels, kernel_size, kernel_size])
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        shared_weight[i] = __ldg(&weight[weight_idx]);
    }
    __syncthreads();

    // Start with bias using read-only cache
    float result = __ldg(&bias[o]);

    // Use tiling for the input channels
    const int TILE_SIZE = 4;
    #pragma unroll
    for (int c_base = 0; c_base < in_channels; c_base += TILE_SIZE) {
        float temp[TILE_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int p = 0; p < kernel_size; ++p) {
            int h_unscaled = out_h + padding - p * dilation;
            if (h_unscaled % stride != 0)
                continue;
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height)
                continue;
            
            #pragma unroll
            for (int q = 0; q < kernel_size; ++q) {
                int w_unscaled = out_w + padding - q * dilation;
                if (w_unscaled % stride != 0)
                    continue;
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width)
                    continue;
                
                #pragma unroll
                for (int offset = 0; offset < TILE_SIZE; ++offset) {
                    int c = c_base + offset;
                    if (c < in_channels) {
                        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                        // Access shared memory for weights. The layout: [in_channels][kernel_size][kernel_size]
                        int weight_shared_idx = (c * kernel_size + p) * kernel_size + q;
                        float in_val = __ldg(&input[input_idx]);
                        temp[offset] += in_val * shared_weight[weight_shared_idx];
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (c_base + i < in_channels) {
                result += temp[i];
            }
        }
    }

    int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
    output[output_idx] = result;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_shared_ldg(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);  // assume square kernel

    // Compute output dimensions
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Configure thread blocks and grid: using a 2D block (16x16) on spatial dimensions,
    // and grid.z covering batch and output channel combinations
    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x,
              (out_height + block.y - 1) / block.y,
              batch_size * out_channels);

    // Compute shared memory size needed (for the weight tile)
    int weight_count = in_channels * kernel_size * kernel_size;
    size_t shared_mem_size = weight_count * sizeof(float);

    conv_transpose2d_forward_kernel_shared_ldg<<<grid, block, shared_mem_size>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      output.data_ptr<float>(),
      batch_size,
      in_channels,
      out_channels,
      in_height,
      in_width,
      kernel_size,
      out_height,
      out_width,
      stride,
      padding,
      dilation);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error in conv_transpose2d_forward_kernel_shared_ldg: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper function to handle optional bias (None case)
torch::Tensor conv_transpose2d_forward_wrapper_shared_ldg(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {

    int out_channels = weight.size(1);
    torch::Tensor bias;
    if (bias_obj.is(pybind11::none())) {
        bias = torch::zeros({out_channels}, weight.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }

    return conv_transpose2d_forward_cuda_shared_ldg(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_shared_ldg,
        "ConvTranspose2d forward using combined shared memory and __ldg optimization (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
