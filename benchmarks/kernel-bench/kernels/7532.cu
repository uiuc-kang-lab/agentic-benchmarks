#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_coalesced_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W,
    const int out_C, const int out_D, const int out_H, const int out_W,
    const int kD, const int kH, const int kW,
    const int sD, const int sH, const int sW,
    const int pD, const int pH, const int pW,
    const int groups
) {
    // Shared memory for weight tiles
    extern __shared__ float shared_weight[];
    
    // Calculate output position
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int out_size = N * out_C * out_D * out_H * out_W;
    
    // Process multiple elements per thread for better arithmetic intensity
    for (int idx = tid; idx < out_size; idx += stride) {
        const int w = idx % out_W;
        const int h = (idx / out_W) % out_H;
        const int d = (idx / (out_W * out_H)) % out_D;
        const int c = (idx / (out_W * out_H * out_D)) % out_C;
        const int n = idx / (out_W * out_H * out_D * out_C);
        
        const int group = c / (out_C / groups);
        const int c_per_group = C / groups;
        const int oc_per_group = out_C / groups;
        const int c_offset = group * c_per_group;
        
        scalar_t sum = 0.0f;
        
        // Load bias if present
        if (bias != nullptr) {
            sum = __ldg(&bias[c]);
        }
        
        // Compute input bounds for this output position
        const int w_start = (w + pW) / sW;
        const int h_start = (h + pH) / sH;
        const int d_start = (d + pD) / sD;
        
        #pragma unroll 2
        for (int ic = 0; ic < c_per_group; ++ic) {
            const int input_c = c_offset + ic;
            
            // Preload weight values into shared memory
            if (threadIdx.x < kD * kH * kW) {
                shared_weight[threadIdx.x] = __ldg(&weight[
                    ((input_c * oc_per_group + (c % oc_per_group)) * kD * kH * kW) + threadIdx.x
                ]);
            }
            __syncthreads();
            
            #pragma unroll 2
            for (int kd = 0; kd < kD; ++kd) {
                const int id = (d + pD - kd) / sD;
                if (id >= 0 && id < D) {
                    #pragma unroll 2
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = (h + pH - kh) / sH;
                        if (ih >= 0 && ih < H) {
                            #pragma unroll 4
                            for (int kw = 0; kw < kW; ++kw) {
                                const int iw = (w + pW - kw) / sW;
                                if (iw >= 0 && iw < W) {
                                    // Coalesced input read
                                    const scalar_t input_val = __ldg(&input[
                                        ((n * C + input_c) * D + id) * H * W +
                                        ih * W + iw
                                    ]);
                                    
                                    // Use shared memory for weight access
                                    const scalar_t weight_val = shared_weight[
                                        (kd * kH + kh) * kW + kw
                                    ];
                                    
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        
        // Coalesced output write
        output[idx] = sum;
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
    // Ensure inputs are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    auto bias_tensor = bias.has_value() ? bias.value().contiguous() : torch::Tensor();
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);
    
    const int out_C = weight.size(1) * groups;
    const int out_D = (D - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
    const int out_H = (H - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_W = (W - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];
    
    auto output = torch::zeros({N, out_C, out_D, out_H, out_W}, input.options());
    
    const int threads = 256;
    const int blocks = std::min(65535, (N * out_C * out_D * out_H * out_W + threads - 1) / threads);
    
    // Shared memory size for weight caching
    const int shared_mem_size = kD * kH * kW * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_coalesced_kernel", ([&] {
        transposed_conv3d_coalesced_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, C, D, H, W,
            out_C, out_D, out_H, out_W,
            kD, kH, kW,
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