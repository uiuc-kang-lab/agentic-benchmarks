#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel_shared_memory(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    extern __shared__ float shared_weight[];
    // Each block is mapped to a single (b, oc) pair
    int block_id = blockIdx.x;
    int b = block_id / OC;
    int oc = block_id % OC;
    
    // Each thread in the block will process one or more output pixels from the (H x W) plane
    int num_pixels = H * W;
    

    float sum = 0.0f;
    
    // Load weights into shared memory
    for (int ic = threadIdx.x; ic < IC; ic += blockDim.x) {
        shared_weight[ic] = weight[oc * IC + ic];
    }
    __syncthreads();
    
    // 1x1 convolution equivalent to matmul over channels
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        sum += x[x_offset] * shared_weight[ic];
    }
    
    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Handle optional bias
    if (threadIdx.x % warpSize == 0) {
        output[index] = bias ? sum + bias[oc] : sum;
    }
}

torch::Tensor forward_cuda_shared_memory(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    // Input validation
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    // Create output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Launch kernel
    const int threads = 256;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    const int shared_memory_size = IC * sizeof(float);
    
    forward_kernel_shared_memory<<<blocks, threads, shared_memory_size>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_shared_memory, "Pointwise 2D convolution forward with shared memory (CUDA)");
}