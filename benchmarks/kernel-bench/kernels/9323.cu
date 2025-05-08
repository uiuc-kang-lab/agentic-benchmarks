#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void conv_transpose1d_batch_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ output_ptr,
    int current_batch,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int oc = blockIdx.x;
    int o = threadIdx.x;

    extern __shared__ float sweight[];
    const int weight_size = in_channels * kernel_size;
    
    for (int idx = threadIdx.x; idx < weight_size; idx += blockDim.x) {
        int ic = idx / kernel_size;
        int k = idx % kernel_size;
        sweight[idx] = weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
    }
    __syncthreads();

    if (o >= output_length) return;
    
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            int x_idx = current_batch * in_channels * input_length + ic * input_length + i;
            sum += x_ptr[x_idx] * sweight[ic * kernel_size + k]; /* FIXED TYPO HERE: x_xdd > x_idx */
        }
    }

    if (bias_ptr) sum += bias_ptr[oc];
    output_ptr[current_batch * out_channels * output_length + oc * output_length + o] = sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    x = x.contiguous();
    weight = weight.contiguous();
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);

    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());
    const float* bias_ptr = bias ? bias->contiguous().data_ptr<float>() : nullptr;
    const size_t shared_mem = in_channels * kernel_size * sizeof(float);

    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads_per_block = (output_length + 255) / 256 * 256;
    for (int b = 0; b < batch_size; ++b) {
        cudaStream_t stream = streams[b % num_streams];
        conv_transpose1d_batch_kernel<<<out_channels, threads_per_block, shared_mem, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            b,
            in_channels,
            out_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "ConvTranspose1D with stream-parallel batches",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}