#include <torch/extension.h>
#include <ATen/ATen.h>

#define BLOCK_SIZE 256

// Each block computes one output element using cooperative threads and shared memory reduction.
__global__ void conv3d_tile_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Use external shared memory for reduction
    extern __shared__ float sdata[];

    // Each block is responsible for one output element
    int eid = blockIdx.x;

    // Decode the flattened output index (eid) into 5D coordinates: [b, oc, od, oh, ow]
    int ow = eid % out_width;
    int tmp = eid / out_width;
    int oh = tmp % out_height;
    tmp /= out_height;
    int od = tmp % out_depth;
    tmp /= out_depth;
    int oc = tmp % out_channels;
    int b  = tmp / out_channels;

    float sum = 0.0f;

    // Determine group and corresponding input channel range
    int group = oc / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;

    // Total number of contributions for the current output element
    int total_contrib = in_channels_per_group * kernel_d * kernel_h * kernel_w;

    // Each thread in the block processes a subset of the summation elements
    for (int i = threadIdx.x; i < total_contrib; i += blockDim.x) {
        // Map linear index i to (ic, kd, kh, kw)
        int kw = i % kernel_w;
        int tmp_i = i / kernel_w;
        int kh = tmp_i % kernel_h;
        tmp_i /= kernel_h;
        int kd = tmp_i % kernel_d;
        int ic = tmp_i / kernel_d;  // ic in [0, in_channels_per_group)

        int in_c = group * in_channels_per_group + ic;

        // Compute the corresponding input coordinate
        int d_in = od * stride - padding + kd * dilation;
        int h_in = oh * stride - padding + kh * dilation;
        int w_in = ow * stride - padding + kw * dilation;

        // Check bounds
        if (d_in >= 0 && d_in < in_depth &&
            h_in >= 0 && h_in < in_height &&
            w_in >= 0 && w_in < in_width) {
            // Calculate input index: input[b, in_c, d_in, h_in, w_in]
            int input_index = ((b * in_channels + in_c) * in_depth + d_in) * in_height * in_width +
                              h_in * in_width + w_in;
            
            // Calculate weight index: weight[oc, ic, kd, kh, kw]
            int weight_index = ((oc * in_channels_per_group + ic) * kernel_d + kd) * kernel_h * kernel_w +
                               kh * kernel_w + kw;
            
            sum += input[input_index] * weight[weight_index];
        }
    }

    // Store partial sum computed by each thread into shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform shared memory reduction to accumulate the partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result to global memory; no atomic needed as only one block computes this output
    if (threadIdx.x == 0) {
        float result = sdata[0];
        if (bias != nullptr) {
            result += bias[oc];
        }
        output[eid] = result;
    }
}

at::Tensor forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
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

    // Extract input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Extract weight dimensions
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Allocate the output tensor
    auto output = at::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    // Total number of output elements
    int total_outputs = batch_size * out_channels * out_depth * out_height * out_width;

    // Launch one thread block per output element
    int threads = BLOCK_SIZE;
    int blocks = total_outputs;
    size_t shared_mem_size = threads * sizeof(float);

    conv3d_tile_kernel<<<blocks, threads, shared_mem_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        out_depth,
        out_height,
        out_width,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward using tiled shared memory reduction (CUDA)");
}
