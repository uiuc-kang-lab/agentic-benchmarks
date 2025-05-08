#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one output element per block by parallelizing the inner convolution reduction
// across the (in_channels * kernel_size) elements using shared memory and warp-level primitives.

__global__ void conv1d_reduce_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation
) {
    // Map each block to a unique output element index (b, oc, o)
    int global_idx = blockIdx.x;  // gridDim.x == B * out_channels * out_size
    int out_per_b = out_channels * out_size;
    int b = global_idx / out_per_b;
    int rem = global_idx % out_per_b;
    int oc = rem / out_size;
    int o = rem % out_size;
    
    // Total number of multiplications for this output
    int M = in_channels * kernel_size;
    float local_sum = 0.0f;
    
    // Each thread in the block processes a subset of the reduction indices
    for (int idx = threadIdx.x; idx < M; idx += blockDim.x) {
        int ic = idx / kernel_size;
        int k  = idx % kernel_size;
        int input_pos = o * stride + k * dilation;
        if (input_pos < in_size) {
            float a = x[b * (in_channels * in_size) + ic * in_size + input_pos];
            float w = weight[oc * (in_channels * kernel_size) + ic * kernel_size + k];
            local_sum += a * w;
        }
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's lane 0 stores its partial sum into shared memory
    extern __shared__ float sdata[];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Number of warps participating in the block
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float sum_total = 0.0f;

    // Let the first few threads (one per warp) load the partial sums and reduce them using a warp-level reduction
    if (threadIdx.x < num_warps) {
        sum_total = sdata[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_total += __shfl_down_sync(mask, sum_total, offset);
        }
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        if (bias != nullptr) {
            sum_total += bias[oc];
        }
        output[b * (out_channels * out_size) + oc * out_size + o] = sum_total;
    }
}

// Forward function exposed via pybind11

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int dilation
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels mismatch");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().device().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), "Bias size mismatch");
    }

    int B = x.size(0);
    int in_channels = x.size(1);
    int in_size = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_size = (in_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    TORCH_CHECK(out_size > 0, "Invalid output size");

    auto output = torch::empty({B, out_channels, out_size}, x.options());
    if (output.numel() == 0) return output;

    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Each block computes one output element
    int total_outputs = B * out_channels * out_size;
    dim3 grid(total_outputs);
    int threads = 256; // Number of threads per block, can be tuned
    int num_warps = (threads + warpSize - 1) / warpSize;
    int shared_mem_size = num_warps * sizeof(float);

    conv1d_reduce_kernel<<<grid, threads, shared_mem_size>>>(
        x_data,
        weight_data,
        bias_data,
        output_data,
        B,
        in_channels,
        in_size,
        out_channels,
        kernel_size,
        out_size,
        stride,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D convolution forward (CUDA) with shared memory and warp-level reduction");
}
