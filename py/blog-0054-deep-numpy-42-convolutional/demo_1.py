import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy import signal


def conv2d(
    image: NDArray[np.floating], filter_kernel: NDArray[np.floating], stride: int = 1
) -> NDArray[np.floating]:
    """
    Perform 2D convolution on an image using a filter kernel.
    Args:
        image: Input image, shape (height, width)
        filter_kernel: Convolution filter, shape (filter_height, filter_width)
        stride: Stride of the convolution operation (default: 1)
    Returns:
        Output feature map after convolution, shape depends on input, filter size, and stride
    """
    # Use scipy.signal.convolve2d with 'valid' mode (no padding)
    # 'valid' mode means output size is reduced based on filter size
    output = signal.convolve2d(
        image, filter_kernel, mode="valid", boundary="fill", fillvalue=0
    )

    # Apply stride by downsampling the output
    if stride > 1:
        output = output[::stride, ::stride]

    return output
