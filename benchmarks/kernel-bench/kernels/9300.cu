#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

template <int BLOCK_THREADS>
__global__ void conv_transp1d_optimized(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ float smem[];
    
    const int output_id = blockIdx.x;
    const int tid = threadIdx.x;

    const int b = output_id / (out_channels * output_length);
    const int rem = output_id % (out_channels * output_length);
    const int oc = rem / output_length;
    const int o = rem % output_length;

    if (b >= batch_size || oc >= out_channels || o >= output_length) return;

    const int total_ops = kernel_size * in_channels;
    float thread_sum = 0.0f;

    for (int idx = tid; idx < total_ops; idx += BLOCK_THREADS) {
        int k = idx / in_channels;
        int ic = idx % in_channels;

        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        thread_sum += x_ptr[b * in_channels * input_length + ic * input_length + i] 
                      * weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
    }

    smem[tid] = thread_sum;
    __syncthreads();

    // Block reduction
    for (int s = BLOCK_THREADS/2; s > 32; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        float val = smem[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0)
            smem[0] = val;
    }

    if (tid == 0) {
        float final = smem[0] + (bias_ptr ? bias_ptr[oc] : 0.0f);
        output_ptr[b * out_channels * output_length + oc * output_length + o] = final;
    }
}

torch::Tensor forward_optimized_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    const int num_output_elements = batch_size * out_channels * output_length;
    const int BLOCK_SIZE = 256;

    conv_transp1d_optimized<BLOCK_SIZE><<<num_output_elements, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_optimized_cuda, "ConvTransposed1D Optimized (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}