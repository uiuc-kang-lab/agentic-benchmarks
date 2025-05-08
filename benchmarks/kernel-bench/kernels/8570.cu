#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Tile dimensions for output spatial block
#define TILE_H 32
#define TILE_W 32

// Helper function to parse int or sequence of ints
inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        int64_t val = obj.cast<int64_t>();
        result.push_back(val);
        result.push_back(val);
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
        if (result.size() == 1) {
            result.push_back(result[0]);
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

// __global__ kernel: Scatter-based transposed convolution using shared memory
// Each block is assigned a unique tile in the output for a specific batch and output channel.
// Within the block, threads iterate over candidate input pixels that may contribute
// to any output in the tile. Their contributions are accumulated into a shared memory
// tile using atomicAdd (shared memory atomics, which are fast). Finally, one thread writes
// the tile (plus bias) to global memory. This minimizes global atomic operations.

__global__ void conv_transposed2d_scatter_tile_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int in_channels_per_group,
    int out_channels_per_group
) {
    // Determine output tile coordinates
    int tile_col_start = blockIdx.x * TILE_W;
    int tile_row_start = blockIdx.y * TILE_H;

    // Combined index: blockIdx.z encodes batch index and output channel
    int combined = blockIdx.z;
    int n = combined / out_channels;
    int oc = combined % out_channels;

    // Check if the tile is within the output bounds
    if (tile_row_start >= out_h || tile_col_start >= out_w) return;

    // Identify group and corresponding input channel range
    int group = oc / out_channels_per_group;
    int start_ic = group * in_channels_per_group;
    int end_ic = start_ic + in_channels_per_group - 1;
    int oc_inner = oc % out_channels_per_group;

    // Allocate shared memory tile for accumulating contributions
    __shared__ float s_tile[TILE_H * TILE_W];

    // Initialize shared memory tile to 0
    int tid = threadIdx.x;
    int tile_size = TILE_H * TILE_W;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        s_tile[i] = 0.0f;
    }
    __syncthreads();

    // Determine candidate input region that can contribute to this output tile
    // For output coordinate: oh = i_h * stride_h - pad_h + kh
    // We want oh in [tile_row_start, tile_row_start + TILE_H)
    // Solve for i_h: i_h >= ceil((tile_row_start + pad_h - (kernel_h - 1)) / stride_h) and
    // i_h <= floor((tile_row_start + TILE_H - 1 + pad_h) / stride_h)
    int i_h_min = (tile_row_start + pad_h - (kernel_h - 1) + stride_h - 1) / stride_h; // ceiling division
    if (i_h_min < 0) i_h_min = 0;
    int i_h_max = (tile_row_start + TILE_H - 1 + pad_h) / stride_h;
    if (i_h_max >= in_h) i_h_max = in_h - 1;

    // Similarly for width: ow = i_w * stride_w - pad_w + kw
    int i_w_min = (tile_col_start + pad_w - (kernel_w - 1) + stride_w - 1) / stride_w;
    if (i_w_min < 0) i_w_min = 0;
    int i_w_max = (tile_col_start + TILE_W - 1 + pad_w) / stride_w;
    if (i_w_max >= in_w) i_w_max = in_w - 1;

    int num_i_h = (i_h_max >= i_h_min) ? (i_h_max - i_h_min + 1) : 0;
    int num_i_w = (i_w_max >= i_w_min) ? (i_w_max - i_w_min + 1) : 0;

    // Total iterations per input channel: over candidate input pixels and kernel elements
    int iter_per_ic = num_i_h * num_i_w * (kernel_h * kernel_w);
    int ic_count = in_channels_per_group;  // Number of input channels in the group
    int total_iterations = ic_count * iter_per_ic;

    // Loop over candidate contributions distributed among threads in the block
    for (int idx = tid; idx < total_iterations; idx += blockDim.x) {
        // Map linear index back to (ic, i_h_offset, i_w_offset, kernel index)
        int ic_offset = idx / iter_per_ic; // which input channel offset in [0, in_channels_per_group)
        int rem = idx % iter_per_ic;
        int iter_per_hw = kernel_h * kernel_w;
        int hw_idx = rem / iter_per_hw; // index over candidate spatial positions
        int kernel_idx = rem % iter_per_hw; // index over kernel elements

        int i_h_off = hw_idx / num_i_w;
        int i_w_off = hw_idx % num_i_w;

        int current_ic = start_ic + ic_offset;
        int i_h = i_h_min + i_h_off;
        int i_w = i_w_min + i_w_off;

        int kh = kernel_idx / kernel_w;
        int kw = kernel_idx % kernel_w;

        // Compute the corresponding output coordinates
        int oh = i_h * stride_h - pad_h + kh;
        int ow = i_w * stride_w - pad_w + kw;

        // Check if the computed output coordinates fall within the tile
        if (oh >= tile_row_start && oh < tile_row_start + TILE_H &&
            ow >= tile_col_start && ow < tile_col_start + TILE_W &&
            oh < out_h && ow < out_w) {

            // Compute flat indices for input and weight
            int input_index = ((n * in_channels + current_ic) * in_h + i_h) * in_w + i_w;
            int weight_index = ((current_ic * out_channels_per_group + oc_inner) * kernel_h + kh) * kernel_w + kw;
            float in_val = input[input_index];
            float w_val = weight[weight_index];
            float contrib = in_val * w_val;

            int s_index = (oh - tile_row_start) * TILE_W + (ow - tile_col_start);
            atomicAdd(&s_tile[s_index], contrib);
        }
    }

    __syncthreads();

    // One thread writes the accumulated shared memory tile to global output
    if (tid == 0) {
        for (int r = 0; r < TILE_H; ++r) {
            int global_oh = tile_row_start + r;
            if (global_oh >= out_h) continue;
            for (int c = 0; c < TILE_W; ++c) {
                int global_ow = tile_col_start + c;
                if (global_ow >= out_w) continue;
                int s_index = r * TILE_W + c;
                int out_index = ((n * out_channels + oc) * out_h + global_oh) * out_w + global_ow;
                float value = s_tile[s_index];
                // Add bias if provided
                if (bias != nullptr) {
                    value += bias[oc];
                }
                output[out_index] = value;
            }
        }
    }
}

// Forward function wrapper: prepares inputs, computes grid dimensions, and launches the kernel

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    // Parse parameters into 2D vector
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    
    int stride_h = stride_vec[0];
    int stride_w = stride_vec[1];
    int pad_h = padding_vec[0];
    int pad_w = padding_vec[1];
    int output_pad_h = output_padding_vec[0];
    int output_pad_w = output_padding_vec[1];

    // Input dimensions: [N, C, H, W]
    int N = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Weight dimensions: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output dimensions for conv_transpose2d
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    auto output_tensor = torch::zeros({N, out_channels, out_h, out_w}, x.options());

    // Channels per group for input
    int in_channels_per_group = in_channels / groups;

    // Set up grid dimensions: each block covers a tile in the (out_h, out_w) plane for one (n, oc)
    int grid_x = (out_w + TILE_W - 1) / TILE_W;
    int grid_y = (out_h + TILE_H - 1) / TILE_H;
    int grid_z = N * out_channels;  // each (n, oc) combination

    dim3 grid(grid_x, grid_y, grid_z);
    int threads = 256;
    size_t shared_mem_size = TILE_H * TILE_W * sizeof(float);

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
    }

    conv_transposed2d_scatter_tile_kernel<<<grid, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr),
        output_tensor.data_ptr<float>(),
        N,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_channels_per_group,
        out_channels_per_group
    );
    cudaDeviceSynchronize();
    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with scatter in shared memory and minimal global atomics",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
