#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__global__ void conv_transpose1d_kernel(
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
    extern __shared__ float shared_weights[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    // Load weights into shared memory
    for (int i = tid; i < in_channels * out_channels * kernel_size; i += blockDim.x) {
        shared_weights[i] = weight_ptr[i];
    }
    __syncthreads();
    
    if (idx >= batch_size * out_channels * output_length) return;
    
    const int b = idx / (out_channels * output_length);
    const int rem = idx % (out_channels * output_length);
    const int oc = rem / output_length;
    const int o = rem % output_length;
    
    float sum = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        const int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        
        const int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            const int x_idx = b * in_channels * input_length + ic * input_length + i;
            const int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
            sum += x_ptr[x_idx] * shared_weights[weight_idx];
        }
    }
    
    if (bias_ptr) {
        sum += bias_ptr[oc];
    }
    
    output_ptr[b * out_channels * output_length + oc * output_length + o] = sum;
}

torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    
    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());
    
    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_length;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    const int shared_mem_size = in_channels * out_channels * kernel_size * sizeof(float);
    
    conv_transpose1d_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}