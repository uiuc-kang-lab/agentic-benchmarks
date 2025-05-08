#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel_atomic_minimization(
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    
    if (index >= total_elements) return;
    
    // Decompose linear index into 4D tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    
    // Allocate shared memory to minimize global memory atomic operations
    extern __shared__ float shared_sum[];
    shared_sum[threadIdx.x] = 0.0f;
    __syncthreads();
    
    // 1x1 convolution equivalent to matmul over channels
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        const int w_offset = oc * IC + ic;
        atomicAdd(&shared_sum[threadIdx.x], x[x_offset] * weight[w_offset]);
    }

    __syncthreads();

    // Reduce shared memory to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_sum[i];
        }
        // Handle optional bias
        output[index] = sum + (bias ? bias[oc] : 0.0f);
    }
}

torch::Tensor forward_cuda_atomic_minimization(
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
    const int shared_mem_size = threads * sizeof(float);
    
    forward_kernel_atomic_minimization<<<blocks, threads, shared_mem_size>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_atomic_minimization, "Pointwise 2D convolution forward with minimized atomic operations (CUDA)");
}