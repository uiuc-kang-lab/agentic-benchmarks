#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes one output element per block by performing a reduction over the
// convolution summation. The total work (Cin * kernel_size * kernel_size) is divided among
// block threads, with shared memory used to sum partial products. Warp-level primitives are
// used in the final reduction stage.

__global__ void conv2d_kernel_shared_reduce(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int Cin,
    int H,
    int W,
    int Cout,
    int K,
    int stride,
    int padding,
    int dilation,
    int outH,
    int outW) {

    // Each block computes one output element
    // Grid mapping: gridDim.x = outW, gridDim.y = outH, gridDim.z = N * Cout
    int ox = blockIdx.x; // output x coordinate
    int oy = blockIdx.y; // output y coordinate
    int linear = blockIdx.z; // combined index for batch and output channel
    int b = linear / Cout;
    int oc = linear % Cout;

    // Total reduction size: over all input channels and kernel elements
    int total_work = Cin * K * K;
    int tid = threadIdx.x;
    float partial = 0.0f;

    // Distribute the reduction work among threads in the block
    for (int idx = tid; idx < total_work; idx += blockDim.x) {
        int cin = idx / (K * K);
        int rem = idx % (K * K);
        int i = rem / K;
        int j = rem % K;
        int y_in = oy * stride - padding + i * dilation;
        int x_in = ox * stride - padding + j * dilation;
        
        if (y_in >= 0 && y_in < H && x_in >= 0 && x_in < W) {
            int input_idx = ((b * Cin + cin) * H + y_in) * W + x_in;
            int weight_idx = ((oc * Cin + cin) * K + i) * K + j;
            partial += input[input_idx] * weight[weight_idx];
        }
    }

    // Allocate shared memory for reduction
    extern __shared__ float smem[];
    smem[tid] = partial;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for final steps
    float sum = smem[tid];
    if (tid < 32) {
        // Unroll warp-level reduction using shuffle instructions
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Thread 0 writes the result
    if (tid == 0) {
        if (bias != nullptr) {
            sum += bias[oc];
        }
        int out_idx = ((b * Cout + oc) * outH + oy) * outW + ox;
        output[out_idx] = sum;
    }
}

// Forward function that sets up grid and block dimensions
// Grid: (outW, outH, N * Cout) and Block: 256 threads for the reduction
// Shared memory size: blockDim.x * sizeof(float)

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    TORCH_CHECK(groups == 1, "Only groups==1 is supported by this custom kernel.");

    int N = x.size(0);
    int Cin = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int Cout = weight.size(0);
    int K = weight.size(2); // Assuming square kernel

    // Compute output dimensions taking dilation into account
    int outH = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::empty({N, Cout, outH, outW}, x.options());

    // Kernel launch configuration:
    // Each block computes one output element
    dim3 grid(outW, outH, N * Cout);
    int threads = 256;

    const float* bias_ptr = (bias.has_value() ? bias.value().data_ptr<float>() : nullptr);

    conv2d_kernel_shared_reduce<<<grid, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N,
        Cin,
        H,
        W,
        Cout,
        K,
        stride,
        padding,
        dilation,
        outH,
        outW);

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function for 2D convolution with shared memory reduction");
}
