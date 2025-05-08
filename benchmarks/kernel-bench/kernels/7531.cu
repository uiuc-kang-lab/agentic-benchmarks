#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void transposed_conv3d_tiled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W,
    const int K, const int T, const int R, const int S,
    const int OUT_D, const int OUT_H, const int OUT_W,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups
) {
    __shared__ scalar_t shared_input[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t shared_weight[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int out_ch_idx = blockIdx.y;
    const int group = out_ch_idx / (K/groups);
    const int in_ch_start = group * (C/groups);
    const int in_ch_end = (group + 1) * (C/groups);
    
    // Calculate output position
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_w = out_idx % OUT_W;
    const int out_h = (out_idx / OUT_W) % OUT_H;
    const int out_d = (out_idx / (OUT_W * OUT_H)) % OUT_D;
    
    if (out_d >= OUT_D || out_h >= OUT_H || out_w >= OUT_W) return;
    
    scalar_t result = 0.0f;
    
    // Process input in tiles
    for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch += TILE_SIZE) {
        for (int kt = 0; kt < T; kt++) {
            for (int kh = 0; kh < R; kh++) {
                for (int kw = 0; kw < S; kw++) {
                    // Calculate corresponding input position
                    const int in_d = (out_d + pad_d - kt) / stride_d;
                    const int in_h = (out_h + pad_h - kh) / stride_h;
                    const int in_w = (out_w + pad_w - kw) / stride_w;
                    
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W &&
                        (out_d + pad_d - kt) % stride_d == 0 &&
                        (out_h + pad_h - kh) % stride_h == 0 &&
                        (out_w + pad_w - kw) % stride_w == 0) {
                        
                        // Load input tile to shared memory
                        if (tid < TILE_SIZE) {
                            const int in_idx = ((batch_idx * C + (in_ch + tid)) * D * H * W) +
                                             (in_d * H * W) + (in_h * W) + in_w;
                            if ((in_ch + tid) < in_ch_end) {
                                shared_input[tid][0] = input[in_idx];
                            }
                        }
                        
                        // Load weight tile to shared memory
                        if (tid < TILE_SIZE) {
                            const int weight_idx = ((in_ch + tid) * K/groups + (out_ch_idx % (K/groups))) * T * R * S +
                                                 (kt * R * S) + (kh * S) + kw;
                            if ((in_ch + tid) < in_ch_end) {
                                shared_weight[tid][0] = weight[weight_idx];
                            }
                        }
                        
                        __syncthreads();
                        
                        // Compute partial results using shared memory
                        for (int i = 0; i < TILE_SIZE && (in_ch + i) < in_ch_end; i++) {
                            result += shared_input[i][0] * shared_weight[i][0];
                        }
                        
                        __syncthreads();
                    }
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        result += bias[out_ch_idx];
    }
    
    // Write result to output
    if (out_d < OUT_D && out_h < OUT_H && out_w < OUT_W) {
        const int out_offset = ((batch_idx * K + out_ch_idx) * OUT_D * OUT_H * OUT_W) +
                             (out_d * OUT_H * OUT_W) + (out_h * OUT_W) + out_w;
        output[out_offset] = result;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();
    auto bias_tensor = bias.has_value() ? bias.value().contiguous() : torch::Tensor();
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    
    const int K = weight.size(1) * groups;
    const int T = weight.size(2);
    const int R = weight.size(3);
    const int S = weight.size(4);
    
    const int OUT_D = (D - 1) * stride[0] - 2 * padding[0] + T + output_padding[0];
    const int OUT_H = (H - 1) * stride[1] - 2 * padding[1] + R + output_padding[1];
    const int OUT_W = (W - 1) * stride[2] - 2 * padding[2] + S + output_padding[2];
    
    auto output = torch::zeros({N, K, OUT_D, OUT_H, OUT_W}, input.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks_x = (OUT_D * OUT_H * OUT_W + threads - 1) / threads;
    const dim3 blocks(blocks_x, K, N);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_tiled_kernel", ([&] {
        transposed_conv3d_tiled_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, C, D, H, W,
            K, T, R, S,
            OUT_D, OUT_H, OUT_W,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            groups
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward optimized",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}