#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const int tid = threadIdx.x;
    const int wid = tid / 32;  // warp ID
    const int lane = tid % 32;  // lane ID within warp
    const int block_stride = blockDim.x * gridDim.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Process multiple elements per thread
    for (int idx = gid; idx < B * OC * H * W; idx += block_stride) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int oc = (idx / (W * H)) % OC;
        const int b = idx / (W * H * OC);

        float sum = 0.0f;

        // Process channels in chunks of 32 (warp size)
        for (int ic_base = 0; ic_base < IC; ic_base += 32) {
            float partial_sum = 0.0f;
            const int ic = ic_base + lane;
            
            if (ic < IC) {
                const int x_offset = b * IC * H * W + ic * H * W + h * W + w;
                const int w_offset = oc * IC + ic;
                partial_sum = x[x_offset] * weight[w_offset];
            }

            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                partial_sum += __shfl_down_sync(__activemask(), partial_sum, offset);
            }

            // First lane has the sum
            if (lane == 0) {
                sum += partial_sum;
            }
        }

        // Only the first lane writes the result
        if (lane == 0) {
            output[idx] = bias ? sum + bias[oc] : sum;
        }
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

    auto output = torch::empty({B, OC, H, W}, x.options());

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    const int threads = 256;
    const int blocks = (B * OC * H * W + threads - 1) / threads;
    
    forward_kernel<<<blocks, threads>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        B, IC, OC, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Pointwise 2D convolution forward (CUDA)");
}