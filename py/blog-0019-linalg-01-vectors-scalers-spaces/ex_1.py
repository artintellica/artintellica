import numpy as np


def vector_operations(u, v, c):
    """
    Perform vector addition and scaling.

    Parameters:
    u (np.ndarray): First input vector
    v (np.ndarray): Second input vector
    c (float): Scalar for scaling the first vector

    Returns:
    tuple: (sum of vectors, scaled first vector)
    """
    # Vector addition
    vector_sum = u + v
    # Scaling the first vector
    scaled_u = c * u
    return vector_sum, scaled_u


# Test the function
u = np.array([4, 1])
v = np.array([-2, 3])
c = 3

sum_result, scaled_result = vector_operations(u, v, c)

print("Vector u:", u)
print("Vector v:", v)
print("Scalar c:", c)
print("u + v =", sum_result)
print("c * u =", scaled_result)
