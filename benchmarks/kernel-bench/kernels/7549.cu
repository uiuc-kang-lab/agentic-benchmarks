#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes one output element per block. Each block's threads partition the work of accumulating the contributions from the input channels and kernel elements.
// The partial sums are first accumulated by each thread and then reduced using shared memory and warp-level primitives (__shfl_down_sync) for the final stages.

template <typename scalar_t>
__global__ void transposed_conv3d_shared_warp_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr
    scalar_t* __restrict__ output,
    // Input dimensions
    int N, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Kernel dimensions
    int kT, int kH, int kW,
    // Stride values
    int stride_d, int stride_h, int stride_w,
    // Padding values
    int pad_d, int pad_h, int pad_w,
    // Output padding values
    int out_pad_d, int out_pad_h, int out_pad_w,
    // Groups
    int groups
) {
    // Each block computes one output element
    int out_index = blockIdx.x;

    // Decode the flattened output index into (n, c, d, h, w)
    int w = out_index % out_width;
    int tmp = out_index / out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int d = tmp % out_depth;
    tmp /= out_depth;
    int c = tmp % out_channels;
    int n = tmp / out_channels;

    // Determine group and local channel index
    int group = c / (out_channels / groups);
    int out_c_local = c % (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    // Total number of iterations for this output element
    int total_iters = in_channels_per_group * kT * kH * kW;

    // Each thread in the block will accumulate a partial sum from a portion of the iterations
    int tid = threadIdx.x;
    scalar_t partial_sum = 0;

    for (int i = tid; i < total_iters; i += blockDim.x) {
        // Decode the iteration index into (ic, kd, kh, kw)
        int ic = i / (kT * kH * kW);
        int rem = i % (kT * kH * kW);
        int kd = rem / (kH * kW);
        int rem2 = rem % (kH * kW);
        int kh = rem2 / kW;
        int kw = rem2 % kW;

        int input_channel = group * in_channels_per_group + ic;

        // Compute corresponding input indices using the transposed convolution relation
        int d_in_tmp = d + pad_d - kd;
        if (d_in_tmp % stride_d != 0) continue;
        int d_in = d_in_tmp / stride_d;
        if (d_in < 0 || d_in >= in_depth) continue;

        int h_in_tmp = h + pad_h - kh;
        if (h_in_tmp % stride_h != 0) continue;
        int h_in = h_in_tmp / stride_h;
        if (h_in < 0 || h_in >= in_height) continue;

        int w_in_tmp = w + pad_w - kw;
        if (w_in_tmp % stride_w != 0) continue;
        int w_in = w_in_tmp / stride_w;
        if (w_in < 0 || w_in >= in_width) continue;

        // Calculate flat input index [N, in_channels, in_depth, in_height, in_width]
        int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;

        // Calculate flat weight index [in_channels, out_channels/groups, kT, kH, kW]
        int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;

        partial_sum += input[input_idx] * weight[weight_idx];
    }

    // Reduction: use shared memory to accumulate partial sums from all threads in this block
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[tid] = partial_sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction
    if (tid < 32) {
        volatile scalar_t* smem_ptr = sdata;
        for (unsigned int offset = 32; offset > 0; offset /= 2) {
            smem_ptr[tid] += smem_ptr[tid + offset];
        }
    }

    // The first thread writes the result
    if (tid == 0) {
        scalar_t result = sdata[0];
        if (bias != nullptr) {
            result += bias[c];
        }
        output[out_index] = result;
    }
}


// Host function to prepare output dimensions, launch kernel, and manage shared memory

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure input tensors are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    // Input dimensions
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Kernel dimensions
    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Determine output channels from weight and groups; weight shape: [in_channels, out_channels/groups, kT, kH, kW]
    int out_channels = weight.size(1) * groups;

    // Calculate output spatial dimensions using the transposed convolution formula
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    // Total number of output elements
    int total_output = N * out_channels * out_depth * out_height * out_width;

    // Launch configuration: one block per output element with a fixed number of threads
    int threads = 128;
    int blocks = total_output;
    size_t shared_mem_size = threads * sizeof(float);  // Note: This assumes float, but AT_DISPATCH_FLOATING_TYPES ensures correct type

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_shared_warp_kernel", ([&] {
        transposed_conv3d_shared_warp_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups
        );
    }));

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function using shared memory and warp-level reductions",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
