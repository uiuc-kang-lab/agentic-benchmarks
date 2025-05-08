#include <torch/extension.h>

__global__ void add_bias_shared_kernel(
    float* output,
    const float* bias,
    int total,
    int C_out,
    int H_out,
    int W_out) {
    extern __shared__ float bias_sm[];
    
    // Load bias values into shared memory
    int tid = threadIdx.x;
    if (tid < C_out) {
        bias_sm[tid] = bias[tid];
    }
    __syncthreads();
    
    // Process elements with shared memory access
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int oc = (index / (H_out * W_out)) % C_out;
    output[index] += bias_sm[oc];
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    auto output = at::conv_transpose2d(
        x, weight, bias,
        {stride, stride},
        {padding, padding},
        {output_padding, output_padding},
        groups
    );

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
        
        int C_out = weight.size(1);
        int total = output.numel();
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;

        add_bias_shared_kernel<<<grid_size, block_size, C_out*sizeof(float)>>>(
            output.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            total, C_out,
            output.size(2),
            output.size(3)
        );
        cudaDeviceSynchronize();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CUDA) - shared mem bias");
}
