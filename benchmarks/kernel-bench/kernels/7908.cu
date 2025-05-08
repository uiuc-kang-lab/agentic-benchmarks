#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory to load the weights for a given output channel once per block.
// Each block is mapped to a tile of output pixels for a specific (batch, out_channel) pair.
// Each thread computes one output element by iterating over all input channels and the kernel window.
// Since each thread writes to a unique output element, no atomic operations on global memory are used.
// Atomic operations would only be used if multiple threads needed to update the same global variable, but here that race is avoided.

__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int K,
    int outH,
    int outW,
    int stride,
    int padding) {

    // Decode blockIdx.z into batch index and output channel
    int b_oc = blockIdx.z; // ranges over N * Cout
    int n = b_oc / Cout;
    int oc = b_oc % Cout;

    // Compute output tile offsets
    int ox0 = blockIdx.x * blockDim.x;
    int oy0 = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ox = ox0 + tx;
    int oy = oy0 + ty;

    // Allocate shared memory for the weight for this output channel
    // Size: Cin * K * K
    extern __shared__ float shared_weight[];
    int total_weight = Cin * K * K;
    int tid = ty * blockDim.x + tx;
    for (int i = tid; i < total_weight; i += blockDim.x * blockDim.y) {
        shared_weight[i] = weight[((oc * Cin) * K * K) + i];
    }
    __syncthreads();

    float sum = 0.0f;
    if (ox < outW && oy < outH) {
        // Loop over all input channels and kernel elements
        for (int c = 0; c < Cin; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int in_y = oy * stride - padding + kh;
                    int in_x = ox * stride - padding + kw;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int input_idx = ((n * Cin + c) * H + in_y) * W + in_x;
                        int w_idx = (c * K + kh) * K + kw;
                        sum += input[input_idx] * shared_weight[w_idx];
                    }
                }
            }
        }
    }

    if (ox < outW && oy < outH) {
        int output_idx = ((n * Cout + oc) * outH + oy) * outW + ox;
        output[output_idx] = sum;
    }
}


// Host function forwarding the convolution operation
// Assumes square input and square kernel. Dilation is not used in this kernel.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation, // dilation is not applied in this kernel
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }
    TORCH_CHECK(groups == 1, "groups != 1 is not supported by this kernel");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // assuming square kernel (K x K)

    int outH = (H + 2 * padding - K) / stride + 1;
    int outW = (W + 2 * padding - K) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Define block dimensions; each thread computes one output element
    dim3 blockDim(16, 16);
    dim3 gridDim((outW + blockDim.x - 1) / blockDim.x,
                 (outH + blockDim.y - 1) / blockDim.y,
                 N * Cout);

    // Allocate shared memory for weights: size is Cin * K * K floats per block
    size_t shared_mem_size = sizeof(float) * Cin * K * K;

    conv2d_shared_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, H, W, Cout, K, outH, outW, stride, padding);

    cudaDeviceSynchronize();

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory for weights");
}
