#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
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
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int warps_per_block = blockDim.x / 32;
    
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= batch_size * out_channels * output_length) return;

    const int b = idx / (out_channels * output_length);
    const int rem = idx % (out_channels * output_length);
    const int oc = rem / output_length;
    const int o = rem % output_length;

    float partial_sum = 0.0f;

    // Load weights into shared memory
    const int weights_per_thread = (in_channels * kernel_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; i++) {
        const int weight_idx = tid + i * blockDim.x;
        if (weight_idx < in_channels * kernel_size) {
            shared_mem[weight_idx] = weight_ptr[oc * in_channels * kernel_size + weight_idx];
        }
    }
    __syncthreads();

    // Compute partial sums using shared memory weights
    for (int k = 0; k < kernel_size; ++k) {
        const int i_pos = (o - padding) + k * dilation;
        if (i_pos % stride != 0) continue;
        
        const int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        const float* x_batch = x_ptr + b * in_channels * input_length;
        for (int ic = 0; ic < in_channels; ++ic) {
            const int x_idx = ic * input_length + i;
            const int weight_idx = ic * kernel_size + k;
            partial_sum += x_batch[x_idx] * shared_mem[weight_idx];
        }
    }

    // Warp-level reduction
    partial_sum = warp_reduce_sum(partial_sum);

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_mem[wid] = partial_sum;
    }
    __syncthreads();

    // First warp reduces results from all warps
    if (wid == 0) {
        partial_sum = (tid < warps_per_block) ? shared_mem[tid] : 0.0f;
        partial_sum = warp_reduce_sum(partial_sum);

        if (lane == 0) {
            if (bias_ptr) {
                partial_sum += bias_ptr[oc];
            }
            output_ptr[b * out_channels * output_length + oc * output_length + o] = partial_sum;
        }
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
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;

    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    int threads_per_block = 256;
    int num_output_elements = batch_size * out_channels * output_length;
    int num_blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;

    // Shared memory size for weights and warp reduction
    int shared_mem_size = max(
        in_channels * kernel_size * sizeof(float),
        (threads_per_block / 32) * sizeof(float)
    );

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