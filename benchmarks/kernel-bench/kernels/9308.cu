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
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    extern __shared__ float shared_weights[];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int oc = blockIdx.x;
    const int b = blockIdx.y;
    const int o = blockIdx.z;
    
    // Load weights into shared memory
    for (int k = lane_id; k < kernel_size; k += 32) {
        for (int ic = 0; ic < in_channels; ++ic) {
            shared_weights[(ic * kernel_size) + k] = weight_ptr[ic * out_channels * kernel_size + oc * kernel_size + k];
        }
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // Calculate valid input positions
    const int i_pos = o * stride + padding;
    
    for (int k = 0; k < kernel_size; ++k) {
        const int input_pos = i_pos - k * dilation;
        if (input_pos < 0 || input_pos >= input_length * stride || (input_pos % stride != 0)) continue;
        
        const int i = input_pos / stride;
        if (i >= input_length) continue;
        
        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ++ic) {
            const float x_val = x_ptr[b * in_channels * input_length + ic * input_length + i];
            const float w_val = shared_weights[ic * kernel_size + k];
            sum += x_val * w_val;
        }
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        output_ptr[b * out_channels * output_length + oc * output_length + o] = sum;
    }
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
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_length = x.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());
    
    const int shared_mem_size = in_channels * kernel_size * sizeof(float);
    
    dim3 grid(out_channels, batch_size, output_length);
    conv_transpose1d_kernel<<<grid, 32, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias ? bias->data_ptr<float>() : nullptr,
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