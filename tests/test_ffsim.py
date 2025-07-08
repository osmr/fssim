import pytest
import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# Import your function from the library
from fssim import multiply_by_two

# Fixture to check CUDA availability
@pytest.fixture(scope="module")
def cuda_available():
    return torch.cuda.is_available()

# -----------------------------------------------------------------------------
# Positive Tests (Happy Path)
# -----------------------------------------------------------------------------

def test_single_element_tensor(cuda_available):
    """Test with a tensor containing a single element."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_tensor = torch.tensor([5.0], dtype=torch.float32, device='cuda')
    output_tensor = multiply_by_two(input_tensor)
    expected_tensor = torch.tensor([10.0], dtype=torch.float32, device='cuda')
    assert torch.equal(output_tensor, expected_tensor)
    assert not torch.equal(input_tensor, expected_tensor), "Input tensor should not have changed."
    assert output_tensor.is_cuda
    assert output_tensor.dtype == torch.float32

def test_multiple_elements_tensor(cuda_available):
    """Test with a tensor containing multiple elements."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda')
    output_tensor = multiply_by_two(input_tensor)
    expected_tensor = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32, device='cuda')
    assert torch.equal(output_tensor, expected_tensor)
    assert not torch.equal(input_tensor, expected_tensor), "Input tensor should not have changed."
    assert output_tensor.is_cuda
    assert output_tensor.dtype == torch.float32

def test_large_tensor(cuda_available):
    """Test with a large tensor to check performance and correctness."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    size = 1000000 # One million elements
    input_tensor = torch.arange(size, dtype=torch.float32, device='cuda')
    output_tensor = multiply_by_two(input_tensor)
    expected_tensor = torch.arange(size, dtype=torch.float32, device='cuda') * 2.0
    assert torch.equal(output_tensor, expected_tensor)
    assert not torch.equal(input_tensor, expected_tensor), "Input tensor should not have changed."
    assert output_tensor.is_cuda
    assert output_tensor.dtype == torch.float32

def test_zero_tensor(cuda_available):
    """Test with a tensor containing zeros."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_tensor = torch.zeros(10, dtype=torch.float32, device='cuda')
    output_tensor = multiply_by_two(input_tensor)
    expected_tensor = torch.zeros(10, dtype=torch.float32, device='cuda')
    assert torch.equal(output_tensor, expected_tensor)
    assert output_tensor.is_cuda
    assert output_tensor.dtype == torch.float32

def test_negative_values_tensor(cuda_available):
    """Test with a tensor containing negative values."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_tensor = torch.tensor([-1.0, -2.5, -0.0], dtype=torch.float32, device='cuda')
    output_tensor = multiply_by_two(input_tensor)
    expected_tensor = torch.tensor([-2.0, -5.0, -0.0], dtype=torch.float32, device='cuda')
    assert torch.equal(output_tensor, expected_tensor)
    assert not torch.equal(input_tensor, expected_tensor), "Input tensor should not have changed."
    assert output_tensor.is_cuda
    assert output_tensor.dtype == torch.float32

def test_different_shapes(cuda_available):
    """Test with tensors of various shapes (1D, 2D, 3D)."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")

    # 1D
    input_1d = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda')
    output_1d = multiply_by_two(input_1d)
    assert torch.equal(output_1d, torch.tensor([2.0, 4.0], dtype=torch.float32, device='cuda'))
    assert input_1d.shape == output_1d.shape

    # 2D
    input_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device='cuda')
    output_2d = multiply_by_two(input_2d)
    assert torch.equal(output_2d, torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32, device='cuda'))
    assert input_2d.shape == output_2d.shape

    # 3D
    input_3d = torch.arange(8, dtype=torch.float32, device='cuda').reshape(2, 2, 2)
    output_3d = multiply_by_two(input_3d)
    expected_3d = torch.arange(8, dtype=torch.float32, device='cuda').reshape(2, 2, 2) * 2.0
    assert torch.equal(output_3d, expected_3d)
    assert input_3d.shape == output_3d.shape

# -----------------------------------------------------------------------------
# Negative Tests (Error Handling)
# -----------------------------------------------------------------------------

def test_cpu_tensor_raises_error(cuda_available):
    """Test that the function raises an error for a CPU tensor."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cpu')
    with pytest.raises(RuntimeError) as excinfo: # Expecting RuntimeError from TORCH_CHECK
        multiply_by_two(input_cpu)
    assert "Tensor must be on CUDA device" in str(excinfo.value)

def test_incorrect_dtype_raises_error(cuda_available):
    """Test that the function raises an error for a tensor with an incorrect data type."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_int = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda')
    with pytest.raises(RuntimeError) as excinfo: # Expecting RuntimeError from TORCH_CHECK
        multiply_by_two(input_int)
    assert "Only float32 is supported" in str(excinfo.value)

def test_empty_tensor(cuda_available):
    """Test with an empty tensor."""
    if not cuda_available:
        pytest.skip("CUDA not available, skipping test.")
    input_empty = torch.empty(0, dtype=torch.float32, device='cuda')
    output_empty = multiply_by_two(input_empty)
    assert torch.equal(output_empty, torch.empty(0, dtype=torch.float32, device='cuda'))
    assert output_empty.is_cuda
    assert output_empty.dtype == torch.float32
    assert output_empty.shape == input_empty.shape
