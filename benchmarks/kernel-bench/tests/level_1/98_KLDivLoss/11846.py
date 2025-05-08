
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
        is_python_module=True,
    )
    return cuda_module

# Issue 1 & 2: Wrong formula and wrong reduction factor.
# For a simple example, we know what KL divergence should be.
# Here we set up a simple test where the input distributions are uniform.
def test_incorrect_kl_div_formula_and_reduction():
    cuda_mod = build_kernel()
    
    # Create a batch of 4 distributions with 8 elements each.
    batch_size = 4
    num_classes = 8
    # uniform predictions and targets (all equal) so that:
    # log(pred) = log(1/num_classes) and exp(log(pred))=1/num_classes.
    predictions = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float32, device='cuda')
    log_predictions = torch.log(predictions)
    targets = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float32, device='cuda')
    
    # Expected KL divergence (per PyTorch definition): 
    # loss = sum(targets * (log(targets) - log_predictions)) / batch_size
    # But since predictions==targets then log(targets)==log_predictions, so expected loss == 0.
    expected = torch.zeros(1, device='cuda', dtype=torch.float32)
    
    # Compute result from our kernel extension.
    result = cuda_mod.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Due to issues 1 and 2, result will likely differ from zero.
    # We check that the kernel result is not close to the correct result.
    assert not torch.allclose(result, expected, atol=1e-5), (
        f"Expected a discrepancy due to wrong formula and reduction: result {result.item()} vs expected {expected.item()}"
    )

# Issue 3: Kernel only supports float32.
def test_input_tensor_type_not_float32():
    cuda_mod = build_kernel()
    
    # Create double precision inputs.
    batch_size = 4
    num_classes = 16
    predictions = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float64, device='cuda')
    log_predictions = torch.log(predictions)
    targets = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float64, device='cuda')
    
    with pytest.raises(RuntimeError):  # Expecting an error because of type mismatch.
        cuda_mod.forward(log_predictions, targets)

# Issue 4: Reduction assumes blockDim.x is a multiple of 32.
# We trigger this by forcing an input size that causes a kernel launch with a block dimension
# not divisible by 32. We can hack this by compiling with a smaller thread count.
def test_block_dim_not_multiple_of_32():
    # Rebuild the module with a custom kernel launcher.
    # For this test, we'll simulate a kernel launch with a block size of 30.
    # To do this, we create a dummy kernel wrapper that launches the kernel with 30 threads per block.
    from torch.utils.cpp_extension import load_inline

    source = r'''
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    __global__ void kldiv_shared_memory_kernel_custom(
        const float* __restrict__ log_predictions,
        const float* __restrict__ targets,
        float* __restrict__ output,
        const int n) {

        extern __shared__ float s_buffer[];
        float* s_logs = s_buffer;
        float* s_targets = &s_buffer[blockDim.x];
        
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        unsigned int bdim = blockDim.x;
        float sum = 0.0f;

        for (unsigned int tile_base = bid * bdim; tile_base < n; tile_base += gridDim.x * bdim) {
            unsigned int load_idx = tile_base + tid;
            if (load_idx < n) {
                s_logs[tid] = log_predictions[load_idx];
                s_targets[tid] = targets[load_idx];
            } else {
                s_logs[tid] = 0.0f;
                s_targets[tid] = 0.0f;
            }
            __syncthreads();

            unsigned int compute_idx = tile_base + tid;
            if (compute_idx < n) {
                sum += expf(s_logs[tid]) - s_targets[tid] * s_logs[tid];
            }
            __syncthreads();
        }

        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        __shared__ float block_sum[32];
        if (tid % 32 == 0) {
            block_sum[tid/32] = sum;
        }
        __syncthreads();

        if (tid < 32) {
            float val = (tid < (bdim + 31)/32) ? block_sum[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(mask, val, offset);
            }
            if (tid == 0) {
                atomicAdd(output, val);
            }
        }
    }

    torch::Tensor kl_div_cuda_forward_custom(
        const torch::Tensor& log_predictions,
        const torch::Tensor& targets) {
        
        const int n = log_predictions.numel();
        auto output = torch::zeros({1}, log_predictions.options());

        // Force blockDim.x to 30 (not a multiple of 32)
        const unsigned int threads = 30;
        const unsigned int blocks = (n + threads - 1) / threads;
        const size_t shared_mem = 2 * threads * sizeof(float);

        kldiv_shared_memory_kernel_custom<<<blocks, threads, shared_mem>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );

        return output / static_cast<float>(n);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("forward_custom", &kl_div_cuda_forward_custom, "KL divergence with shared memory optimization and custom blockDim");
    }
    '''
    cuda_mod_custom = load_inline(name="kl_div_custom_module",
                                  cpp_sources="",
                                  cuda_sources=source,
                                  functions=["forward_custom"],
                                  extra_cuda_cflags=["-O3", "--use_fast_math"],
                                  verbose=True)
    
    # Use an arbitrary input with size not a multiple of 30.
    batch_size = 4
    num_classes = 37  # Chosen so that total elements ((batch_size*num_classes)=148) not divisible by 30.
    predictions = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float32, device='cuda')
    log_predictions = torch.log(predictions)
    targets = torch.full((batch_size, num_classes), 1.0 / num_classes, dtype=torch.float32, device='cuda')
    
    # Expected KL divergence is 0 as in the uniform case.
    expected = torch.zeros(1, device='cuda', dtype=torch.float32)
    result = cuda_mod_custom.forward_custom(log_predictions, targets)
    torch.cuda.synchronize()
    
    # The reduction with a blockDim not multiple of 32 is likely faulty.
    # We check that the result deviates from the expected 0.
    assert not torch.allclose(result, expected, atol=1e-5), (
        f"Expected a discrepancy due to reduction issues with blockDim not multiple of 32: result {result.item()} vs expected {expected.item()}"
    )
    
if __name__ == '__main__':
    pytest.main([__file__])
