#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Each warp computes one output element using warp-level reduction
__global__ void warp_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int IC,
    int OC,
    int H,
    int W
) {
    // Each warp handles one output element
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize; // one warp per output element
    int lane = threadIdx.x % warpSize;

    int total_outputs = B * OC * H * W;
    if (warp_id >= total_outputs) return;

    // Decode warp_id into 4D output coordinates: b, oc, h, w
    int tmp = warp_id;
    int w_idx = tmp % W;
    tmp /= W;
    int h_idx = tmp % H;
    tmp /= H;
    int oc_idx = tmp % OC;
    tmp /= OC;
    int b_idx = tmp;  

    float sum = 0.0f;
    // Each lane computes partial sum for a portion of the input channels
    for (int ic = lane; ic < IC; ic += warpSize) {
        int x_offset = b_idx * IC * H * W + ic * H * W + h_idx * W + w_idx;
        int w_offset = oc_idx * IC + ic;
        sum += x[x_offset] * weight[w_offset];
    }

    // Warp-level reduction using __shfl_down_sync
    // Use a full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first lane writes the result
    if (lane == 0) {
        if (bias != nullptr) {
            sum += bias[oc_idx];
        }
        int out_offset = b_idx * OC * H * W + oc_idx * H * W + h_idx * W + w_idx;
        output[out_offset] = sum;
    }
}


torch::Tensor forward_cuda(
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

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    // Each warp computes one output element.
    int total_outputs = B * OC * H * W;
    // Total threads: one warp per output element, each warp has 32 threads
    int total_threads = total_outputs * warpSize;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    warp_forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Warp-level optimized pointwise 2D convolution forward (CUDA)");
}
