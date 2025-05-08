#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes 3D convolution output elements cooperatively.
// Each output element is computed by a group of THREADS_PER_OUTPUT threads.
// They split the reduction over the kernel/in_channel dimension and then use shared memory
// followed by warp-level reduction (__shfl_down_sync) for the final summation.

// Constant: number of threads used per output element
#define THREADS_PER_OUTPUT 64

// The cooperative convolution kernel
template <typename scalar_t>
__global__ void conv3d_shared_warp_red_kernel(
    const scalar_t * __restrict__ input,
    const scalar_t * __restrict__ weight,
    scalar_t * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_d,
    int in_h,
    int in_w,
    int out_channels,
    int out_d,
    int out_h,
    int out_w,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_d * out_h * out_w;

    // Each group of THREADS_PER_OUTPUT threads computes one output element
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int group_idx = global_thread_id / THREADS_PER_OUTPUT;   // which output element
    int local_id = global_thread_id % THREADS_PER_OUTPUT;      // thread index within the group

    if (group_idx >= total_outputs) return;  // out-of-bound check

    // Recover output indices from linear group_idx (assuming output tensor shape [b, oc, od, oh, ow])
    int tmp = group_idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int od = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b  = tmp;  // remaining is batch index

    // Determine group and per-group channel counts
    int out_channels_per_group = out_channels / groups;  // assumed divisible
    int in_channels_per_group = in_channels / groups;
    int group = oc / out_channels_per_group;  // which group the output channel belongs to

    // The reduction dimension size: over in_channels_per_group and kernel volume
    int reduction_size = in_channels_per_group * kernel_d * kernel_h * kernel_w;

    // Each thread in the group accumulates a partial sum over a subset of the reduction dimension
    scalar_t partial_sum = 0;
    for (int r = local_id; r < reduction_size; r += THREADS_PER_OUTPUT) {
        // Map r to sub-indices: ic, kd, kh, kw
        int ic = r / (kernel_d * kernel_h * kernel_w);
        int rem = r % (kernel_d * kernel_h * kernel_w);
        int kd = rem / (kernel_h * kernel_w);
        rem = rem % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        // Compute the actual input channel
        int input_channel = group * in_channels_per_group + ic;

        // Compute the corresponding input spatial coordinates
        int id = od * stride - padding + kd * dilation;
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;

        // Check bounds
        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            // Compute input index: [b, input_channel, id, ih, iw]
            int input_idx = (((b * in_channels + input_channel) * in_d + id) * in_h + ih) * in_w + iw;
            // Compute weight index: weight shape [out_channels, in_channels_per_group, kernel_d, kernel_h, kernel_w]
            int weight_idx = (((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw + group * out_channels_per_group * in_channels_per_group * kernel_d * kernel_h * kernel_w;
            partial_sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Use shared memory for intra-block reduction: each thread writes its partial sum
    extern __shared__ char smem[]; // dynamically-allocated shared memory
    scalar_t *sdata = reinterpret_cast<scalar_t*>(smem);
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Identify the start index of the current group's partial sums in shared memory
    int group_in_block = threadIdx.x / THREADS_PER_OUTPUT; // which output computed by this block
    int group_offset = group_in_block * THREADS_PER_OUTPUT;

    // Perform tree reduction in shared memory for THREADS_PER_OUTPUT elements
    // First, reduce from 64 to 32 if applicable
    if (THREADS_PER_OUTPUT >= 64) {
        if (local_id < 32) {
            sdata[group_offset + local_id] += sdata[group_offset + local_id + 32];
        }
        __syncthreads();
    }

    // Now use warp-level primitives for the final reduction.
    if (local_id < 32) {
        scalar_t sum_val = sdata[group_offset + local_id];
        // Unroll warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (local_id == 0) {
            // Write the final result for this output element
            output[group_idx] = sum_val;
        }
    }
}

// Simple kernel for adding bias: one thread per output element
template <typename scalar_t>
__global__ void add_bias_kernel(
    scalar_t * __restrict__ output,
    const scalar_t * __restrict__ bias,
    int total_outputs,
    int out_w,
    int out_h,
    int out_d,
    int out_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_outputs) {
        int tmp = idx;
        int ow = tmp % out_w; tmp /= out_w;
        int oh = tmp % out_h; tmp /= out_h;
        int od = tmp % out_d; tmp /= out_d;
        int oc = tmp % out_channels;  // bias is per channel
        output[idx] += bias[oc];
    }
}

// Host forward function: sets up convolution parameters and launches the cooperative kernel
at::Tensor forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    auto bias = bias_opt.value_or(at::Tensor());
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    }

    // Input dimensions: [batch, in_channels, in_d, in_h, in_w]
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);

    // Weight dimensions: [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Calculate output dimensions using the standard convolution formula
    int out_d = (in_d + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto options = input.options();
    auto output = at::empty({batch_size, out_channels, out_d, out_h, out_w}, options);

    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_d * out_h * out_w;

    // Each output element is computed by THREADS_PER_OUTPUT threads
    // Total threads needed = total_outputs * THREADS_PER_OUTPUT
    int total_threads = total_outputs * THREADS_PER_OUTPUT;
    int threads_per_block = 256; // must be a multiple of THREADS_PER_OUTPUT
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Shared memory size per block in bytes
    size_t shared_mem_size = threads_per_block * sizeof(float);  // works for float; for double it adjusts automatically

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_shared_warp_red_forward_cuda", ([&] {
        conv3d_shared_warp_red_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_d,
            in_h,
            in_w,
            out_channels,
            out_d,
            out_h,
            out_w,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            groups
        );
        
        // If bias is defined, launch a simple kernel to add bias
        if (bias.defined()) {
            int bias_threads = 256;
            int bias_blocks = (total_outputs + bias_threads - 1) / bias_threads;
            add_bias_kernel<scalar_t><<<bias_blocks, bias_threads>>>(
                output.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                total_outputs,
                out_w,
                out_h,
                out_d,
                out_channels
            );
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using cooperative reduction with shared memory and warp-level primitives (CUDA)");
}
