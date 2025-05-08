#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute tensor coordinates from linear index
__device__ inline void compute_coords(int index, int W, int H, int OC, int &b, int &oc, int &h, int &w) {
    w  = index % W;
    h  = (index / W) % H;
    oc = (index / (W * H)) % OC;
    b  = index / (W * H * OC);
}

// Device function to compute convolution for one element
__device__ inline float compute_element(
    const float* x,
    const float* weight,
    int b,
    int oc,
    int h,
    int w,
    int IC,
    int H,
    int W
) {
    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        int x_offset = b * IC * H * W + ic * H * W + h * W + w;
        int w_offset = oc * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }
    return sum;
}

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = B * OC * H * W;
    
    for (int index = tid; index < total_elements; index += stride) {
        int b, oc, h, w;
        compute_coords(index, W, H, OC, b, oc, h, w);
        float sum = compute_element(x, weight, b, oc, h, w, IC, H, W);
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[oc];
        }
        
        output[index] = sum;
    }
}


// CUDA forward function
torch::Tensor forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (OC, IC, 1, 1)");
    if (bias) {
        TORCH_CHECK(bias->is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias->dim() == 1, "Bias must be 1D");
    }

    int B  = x.size(0);
    int IC = x.size(1);
    int H  = x.size(2);
    int W  = x.size(3);
    int OC = weight.size(0);
    
    TORCH_CHECK(weight.size(1) == IC, "Channel mismatch between x and weight");
    TORCH_CHECK(weight.size(2) == 1 && weight.size(3) == 1, "Kernel must be 1x1");
    if (bias) {
        TORCH_CHECK(bias->size(0) == OC, "Bias/out channel mismatch");
    }

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    int total = B * OC * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Modular pointwise 2D convolution forward (CUDA)");
}
