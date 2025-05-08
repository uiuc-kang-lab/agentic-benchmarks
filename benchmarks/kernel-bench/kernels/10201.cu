#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel: loads the weight matrix into shared memory to reduce global memory accesses
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
    // Allocate shared memory for the weight matrix: size (OC x IC)
    extern __shared__ float s_weight[];
    int total_weight = OC * IC;
    // Each thread loads part of the weight matrix from global memory into shared memory
    for (int idx = threadIdx.x; idx < total_weight; idx += blockDim.x) {
        s_weight[idx] = weight[idx];
    }
    // Synchronize to ensure shared memory load completes before computation
    __syncthreads();

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = B * OC * H * W;
    if (index >= total_elements) return;

    // Decompose the linear index into 4D tensor coordinates
    int w = index % W;
    int h = (index / W) % H;
    int oc = (index / (W * H)) % OC;
    int b = index / (W * H * OC);

    float sum = 0.0f;
    // Perform the 1x1 convolution (dot product over input channels) using shared memory for weights
    for (int ic = 0; ic < IC; ++ic) {
        const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        const int weight_index = oc * IC + ic;
        sum += x[x_offset] * s_weight[weight_index];
    }
    
    // If bias exists, add it
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[index] = sum;
}


// Host function which launches the CUDA kernel
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be a CUDA tensor");
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

    // Create the output tensor
    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int total_elements = B * OC * H * W;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Allocate shared memory for the weight matrix: OC * IC floats
    const int shared_mem_size = OC * IC * sizeof(float);
    
    forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}
