#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: each block computes one output element using cooperative reduction.
__global__ void forward_kernel(
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
    // Each block corresponds to one output element. Decompose blockIdx.x into (b, oc, h, w).
    int out_index = blockIdx.x;
    int w = out_index % W;
    int temp = out_index / W;
    int h = temp % H;
    temp = temp / H;
    int oc = temp % OC;
    int b = temp / OC;

    // Each thread computes a partial sum over a subset of the input channels.
    float partial_sum = 0.0f;
    for (int ic = threadIdx.x; ic < IC; ic += blockDim.x) {
        int x_index = b * (IC * H * W) + ic * (H * W) + h * W + w;
        int w_index = oc * IC + ic; // weight shape is (OC, IC, 1, 1)
        partial_sum += x[x_index] * weight[w_index];
    }

    // Use shared memory for block-level reduction.
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Initialize unused shared memory elements to zero to avoid undefined behavior
    if (threadIdx.x >= IC) {
        sdata[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Iterative reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the final result to global memory; no global atomics needed as each block computes a unique output element.
    if (threadIdx.x == 0) {
        float sum = sdata[0];
        if (bias) {
            sum += bias[oc];
        }
        output[out_index] = sum;
    }
}

// CUDA interface for the optimized kernel
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "Bias must be 1D");
    }

    const int B = x.size(0);
    const int IC = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int OC = weight.size(0);

    TORCH_CHECK(weight.size(1) == IC, "Input/output channel mismatch");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().size(0) == OC, "Bias/out channel mismatch");
    }

    // Create output tensor.
    auto output = torch::empty({B, OC, H, W}, x.options());

    // Get raw pointers.
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Each block handles one output element of size (B, OC, H, W).
    int total_output_elements = B * OC * H * W;
    const int threads = 256;
    int blocks = total_output_elements;
    size_t shared_memory = threads * sizeof(float);

    forward_kernel<<<blocks, threads, shared_memory>>>(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        B, IC, OC, H, W
    );

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized Pointwise 2D convolution forward (CUDA)");
}
