
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load
import os
import tempfile

# Helper function to rebuild the extension from kernel.cu in a temporary directory
def build_kernel():
    temp_dir = tempfile.mkdtemp()
    cu_file = os.path.join(temp_dir, "kernel.cu")
    # Write the provided kernel code into the file. (In practice, you would have the actual file.)
    kernel_code = r'''
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <pybind11/pybind11.h>
    #include <vector>

    namespace py = pybind11;

    // Use constant memory for kernel weights (up to 64KB)
    __constant__ float c_weight[16384];

    // Kernel that processes one output element per thread.
    __global__ void conv_transpose2d_forward_kernel_coalesced(
        const float* __restrict__ input,
        const float* __restrict__ bias,
        float* __restrict__ output,
        const int N,
        const int C_in,
        const int H_in,
        const int W_in,
        const int C_out,
        const int H_out,
        const int W_out,
        const int kH,
        const int kW,
        const int sH,
        const int sW,
        const int pH,
        const int pW
    ) {
        int ow = blockIdx.x * blockDim.x + threadIdx.x;
        int oh = blockIdx.y * blockDim.y + threadIdx.y;
        int linear = blockIdx.z;
        int n  = linear / C_out;
        int oc = linear % C_out;

        if (ow < W_out && oh < H_out) {
            float sum = 0.0f;
            for (int ic = 0; ic < C_in; ++ic) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int i_val = oh + pH - kh;
                        int j_val = ow + pW - kw;
                        if ((i_val % sH == 0) && (j_val % sW == 0)) {
                            int i_in = i_val / sH;
                            int j_in = j_val / sW;
                            if (i_in >= 0 && i_in < H_in && j_in >= 0 && j_in < W_in) {
                                int input_idx = ((n * C_in + ic) * H_in + i_in) * W_in + j_in;
                                int weight_idx = ((ic * C_out + oc) * kH + kh) * kW + kw;
                                sum += input[input_idx] * c_weight[weight_idx];
                            }
                        }
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int out_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
            output[out_idx] = sum;
        }
    }

    torch::Tensor conv_transpose2d_forward(
        torch::Tensor x,
        torch::Tensor weight,
        py::object bias_obj,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding
    ) {
        int weight_size = weight.numel() * sizeof(float);
        const int max_const_size = 64 * 1024; // 64KB
        if (weight_size > max_const_size) {
            c10::optional<torch::Tensor> bias = c10::nullopt;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            return at::conv_transpose2d(x, weight, bias, stride, padding);
        }

        cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), weight_size);

        torch::Tensor bias;
        const float* bias_ptr = nullptr;
        if (!bias_obj.is_none()) {
            bias = bias_obj.cast<torch::Tensor>();
            bias_ptr = bias.data_ptr<float>();
        }

        const int N = x.size(0);
        const int C_in = x.size(1);
        const int H_in = x.size(2);
        const int W_in = x.size(3);

        const int C_out = weight.size(1);
        const int kH = weight.size(2);
        const int kW = weight.size(3);

        const int sH = stride[0];
        const int sW = stride[1];
        const int pH = padding[0];
        const int pW = padding[1];

        const int H_out = (H_in - 1) * sH - 2 * pH + kH;
        const int W_out = (W_in - 1) * sW - 2 * pW + kW;

        auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

        dim3 block(16, 16);
        dim3 grid((W_out + block.x - 1) / block.x,
                  (H_out + block.y - 1) / block.y,
                  N * C_out);

        conv_transpose2d_forward_kernel_coalesced<<<grid, block>>>(
            x.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            kH, kW,
            sH, sW,
            pH, pW
        );

        return output;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward with coalesced memory accesses",
              py::arg("x"),
              py::arg("weight"),
              py::arg("bias") = py::none(),
              py::arg("stride"),
              py::arg("padding"));
    }
    '''
    with open(cu_file, "w") as f:
        f.write(kernel_code)
    module = load(
        name="test_kernel",
        sources=[cu_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Type inflexibility – the kernel only supports float32.
def test_dtype_mismatch():
    mod = build_kernel()
    # Create input and weight tensors in double precision.
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64, device="cuda")
    weight = torch.randn(3, 4, 3, 3, dtype=torch.float64, device="cuda")
    bias = torch.randn(4, dtype=torch.float64, device="cuda")
    # The kernel expects float32. Expect an error when calling forward.
    with pytest.raises(RuntimeError):
        mod.forward(x, weight, bias, [1, 1], [1, 1])

# Issue 5: Non-contiguous weight tensor handling.
def test_noncontiguous_weight():
    mod = build_kernel()
    # Create a contiguous weight and then make it non-contiguous via a transpose.
    x = torch.randn(1, 3, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(3, 4, 3, 3, device="cuda", dtype=torch.float32)
    # Make a non-contiguous version (simulate a permutation which is not contiguous).
    weight_nc = weight.permute(1, 0, 2, 3)
    # Our kernel expects weight to be laid out as (in_channels, out_channels, kH, kW).
    # Passing a non-contiguous tensor may lead to the wrong weight values.
    output_kernel = mod.forward(x, weight_nc, None, [1, 1], [1, 1])
    # Compute reference using PyTorch's conv_transpose2d with the expected weight layout.
    # For fairness, convert the noncontiguous weight to contiguous via .contiguous()
    ref = nn.ConvTranspose2d(3, 4, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # Manually set the reference weight (reordering the weight_nc back to the expected layout)
    ref.weight.data.copy_(weight_nc.permute(1, 0, 2, 3))
    output_ref = ref(x)
    # Expect the outputs to differ because the kernel does not handle non-contiguous weights properly.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel output matches reference despite non‐contiguous weight, but it should be wrong."

# Issue 4: Potential grid dimension overflow.
def test_grid_dimension_overflow():
    mod = build_kernel()
    # Choose parameters such that N*C_out exceeds the typical maximum gridDim.z (65535)
    # Use very small weights so that constant memory usage is low.
    # For example, in_channels=1, out_channels=100, and batch size 700 => 700*100 = 70000 > 65535.
    N = 700
    C_in = 1
    C_out = 100
    H_in = 5
    W_in = 5
    kH = 1
    kW = 1
    sH, sW = 1, 1
    pH, pW = 0, 0

    x = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kH, kW, device="cuda", dtype=torch.float32)
    bias = None

    # This should trigger a grid dimension overflow error on architectures with limited gridDim.z.
    with pytest.raises(cuda_error:= (RuntimeError)):
        mod.forward(x, weight, bias, [sH, sW], [pH, pW])
