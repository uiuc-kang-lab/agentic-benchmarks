
import pytest
import torch
from torch.utils.cpp_extension import load
import os
import tempfile

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    # Write the kernel code to a temporary file (simulate the file kernel.cu)
    kernel_code = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to compute GELU activation for a scalar value
__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Device function to compute GELU activation for a float4 vector
__device__ __forceinline__ float4 compute_gelu_vector(const float4 v) {
    float4 out;
    out.x = compute_gelu(v.x);
    out.y = compute_gelu(v.y);
    out.z = compute_gelu(v.z);
    out.w = compute_gelu(v.w);
    return out;
}

// Kernel to process input in vectorized float4 chunks
__global__ void gelu_kernel_vector(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        // Load data using read-only cache
        float4 input = __ldg(&x[idx]);
        // Apply the modular GELU operation
        float4 output = compute_gelu_vector(input);
        y[idx] = output;
    }
}

// Fallback scalar kernel for remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = compute_gelu(x[idx]);
    }
}

// Forward function exposed to Python
torch::Tensor gelu_forward(torch::Tensor x) {
    // Issue 1: No explicit check for input dtype, but we assume float32.
    TORCH_CHECK(x.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process the bulk of data using vectorized operations
    int vec_size = n / 4;  // number of float4 vectors
    int remainder = n % 4;

    // Use 128 threads per block for better occupancy balance
    const int threads = 128;
    if (vec_size > 0) {
        int blocks = (vec_size + threads - 1) / threads;
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_vector<<<blocks, threads>>>(x_vec, y_vec, vec_size);
    }

    // Process any remaining elements with the scalar kernel
    if (remainder > 0) {
        int offset = vec_size * 4;
        int blocks = (remainder + threads - 1) / threads;
        gelu_kernel_scalar<<<blocks, threads>>>(x.data_ptr<float>() + offset, y.data_ptr<float>() + offset, remainder);
    }
    // Optionally check for launch errors (Issue 3)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Modular GELU CUDA implementation");
}
    '''
    temp_dir = tempfile.mkdtemp()
    kernel_path = os.path.join(temp_dir, "kernel.cu")
    with open(kernel_path, "w") as f:
        f.write(kernel_code)
    cuda_module = load(
        name="gelu_cuda",
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Test using a non-float32 tensor (e.g. float64)
def test_dtype_check():
    my_kernel = build_kernel()
    x = torch.randn(100, device="cuda", dtype=torch.float64)  # wrong dtype
    with pytest.raises(RuntimeError, match="Input tensor must be of type float32"):
        y = my_kernel.forward(x)
        torch.cuda.synchronize()

# Issue 2: Test using a non-contiguous tensor.
def test_contiguity_check():
    my_kernel = build_kernel()
    # Create a contiguous tensor and then make a non-contiguous view
    x_contig = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    x_non_contig = x_contig.t()  # transpose makes it non-contiguous
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        y = my_kernel.forward(x_non_contig)
        torch.cuda.synchronize()

# Issue 3: Test potential misalignment. We simulate misaligned memory by creating a tensor
# that is a non-zero offset subset of a larger tensor. This does not guarantee misalignment on all devices,
# but it can serve as a test case in environments where the tensor storage is not naturally aligned for float4.
def test_misaligned_tensor():
    my_kernel = build_kernel()
    # Create a larger tensor and take a narrow slice that might not be 16-byte aligned.
    x_large = torch.randn(1025, device="cuda", dtype=torch.float32)
    x_offset = x_large.narrow(0, 1, 1024)  # starting at offset 1
    # Although x_offset is contiguous, its underlying pointer might be misaligned for float4.
    # We compute GELU using our kernel and compare against PyTorch's implementation.
    y_kernel = my_kernel.forward(x_offset)
    y_ref = 0.5 * x_offset * (1.0 + torch.tanh((x_offset + 0.044715 * x_offset.pow(3)) * 0.7978845608))
    torch.cuda.synchronize()
    # Allow a small tolerance for fp arithmetic differences.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel results differ when using a misaligned tensor."

if __name__ == "__main__":
    pytest.main([__file__])
