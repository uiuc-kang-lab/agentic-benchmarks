#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper function to compute output length based on convolution parameters
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// This kernel distributes the inner work for each output element evenly among all threads in a block.
// Each block computes one output element identified by (batch, out_channel, output_position).
// Within the block, the loop over kernel positions and input channels (flattened as total_iters = in_channels * kernel_size)
// is divided among the threads. A shared memory reduction then combines partial sums from all threads, ensuring an even workload distribution.
__global__ void even_workload_conv_transpose1d_kernel(
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
    // Each block computes one output element
    int out_idx = blockIdx.x;
    if (out_idx >= batch_size * out_channels * output_length) return;

    // Decode out_idx into batch (b), output channel (oc) and output position (o)
    int b = out_idx / (out_channels * output_length);
    int rem = out_idx % (out_channels * output_length);
    int oc = rem / output_length;
    int o = rem % output_length;

    // Flatten the two inner loops: over kernel positions (k) and input channels (ic)
    int total_iters = in_channels * kernel_size;
    float partial_sum = 0.0f;
    
    // Each thread handles a subset of the iterations
    for (int idx = threadIdx.x; idx < total_iters; idx += blockDim.x) {
        int k = idx / in_channels;
        int ic = idx % in_channels;
        int i_pos = o + padding - k * dilation;
        if (i_pos % stride != 0) continue;
        int i = i_pos / stride;
        if (i < 0 || i >= input_length) continue;

        int x_index = b * (in_channels * input_length) + ic * input_length + i;
        int w_index = ic * (out_channels * kernel_size) + oc * kernel_size + k;
        partial_sum += x_ptr[x_index] * weight_ptr[w_index];
    }

    // Use shared memory to reduce partial sums across threads in the block
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduction in shared memory (assumes blockDim.x is a power of 2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = sdata[0];
        if (bias_ptr) {
            sum += bias_ptr[oc];
        }
        output_ptr[out_idx] = sum;
    }
}

// CUDA forward function wrapped for pybind11 binding
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
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    // Total number of output elements
    int total_outputs = batch_size * out_channels * output_length;
    // Launch one block per output element; each block uses a fixed number of threads
    int threads_per_block = 128; // Tuning parameter
    int blocks = total_outputs;

    // Shared memory size per block
    size_t shared_mem_size = threads_per_block * sizeof(float);

    even_workload_conv_transpose1d_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("forward", &forward_cuda, "ConvTranspose1D forward (CUDA) with even workload distribution",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}
