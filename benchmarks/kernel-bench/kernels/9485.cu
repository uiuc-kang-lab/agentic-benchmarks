#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel that computes one output element per block using intra-block reduction with shared memory and warp-level primitives.

__global__ void conv_transpose2d_forward_kernel_reduction(
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

    // Grid mapping: each block computes one output element.
    // blockIdx.x: output width coordinate, blockIdx.y: output height coordinate,
    // blockIdx.z: combined index for (batch, out_channel): b * out_channels + o
    int out_w = blockIdx.x;
    int out_h = blockIdx.y;
    int bo_idx = blockIdx.z;
    int b = bo_idx / out_channels;
    int o = bo_idx % out_channels;

    // Total number of summation elements over in_channels and kernel spatial positions
    int total_elements = in_channels * kernel_size * kernel_size;
    float partial = 0.0f;

    // Each thread processes a portion of the reduction.
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int c = idx / (kernel_size * kernel_size);
        int rem = idx % (kernel_size * kernel_size);
        int p = rem / kernel_size;
        int q = rem % kernel_size;

        int h_unscaled = out_h + padding - p * dilation;
        if (h_unscaled % stride != 0)
            continue;
        int h_in = h_unscaled / stride;
        if (h_in < 0 || h_in >= in_height)
            continue;

        int w_unscaled = out_w + padding - q * dilation;
        if (w_unscaled % stride != 0)
            continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width)
            continue;

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        partial += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
    }

    // Intra-warp reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(mask, partial, offset);
    }

    // Shared memory reduction across warps
    extern __shared__ float sdata[];  // size = (blockDim.x/32) * sizeof(float)
    int lane = threadIdx.x & 31;         // thread index within warp
    int warpId = threadIdx.x >> 5;         // warp index within block

    if (lane == 0) {
        sdata[warpId] = partial;
    }
    __syncthreads();

    // Let the first warp finalize the reduction
    if (warpId == 0) {
        // Load warp sums into 'partial' for threads within first warp
        partial = (threadIdx.x < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        if (lane == 0) {
            float out_val = __ldg(&bias[o]) + partial;
            int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
            output[output_idx] = out_val;
        }
    }
}

// CUDA launcher function

torch::Tensor conv_transpose2d_forward_cuda_reduction(
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
    int kernel_size = weight.size(2); // assuming square kernel

    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Grid: one block per output element
    dim3 blocks(out_width, out_height, batch_size * out_channels);
    int threads = 128;  // Number of threads per block
    size_t shared_mem_size = ((threads + 31) / 32) * sizeof(float);

    conv_transpose2d_forward_kernel_reduction<<<blocks, threads, shared_mem_size>>>(
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
        printf("Error in conv_transpose2d_forward_kernel_reduction: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Wrapper function to handle the possibility of bias being None

torch::Tensor conv_transpose2d_forward_wrapper_reduction(
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

    return conv_transpose2d_forward_cuda_reduction(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward_wrapper_reduction,
          "ConvTranspose2d forward with shared memory reduction and warp-level primitives (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
}
