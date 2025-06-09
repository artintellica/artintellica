import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List

def momentum_update(velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Perform a momentum-based update on the velocity and return the parameter update.
    Args:
        velocity: Current velocity array, same shape as gradient (initialized to 0)
        gradient: Current gradient of the loss with respect to the parameter
        mu: Momentum coefficient (e.g., 0.9)
        lr: Learning rate (e.g., 0.1)
    Returns:
        Tuple of (updated_velocity, parameter_update):
        - updated_velocity: New velocity after momentum update
        - parameter_update: Update to apply to the parameter (e.g., W += parameter_update)
    """
    updated_velocity = mu * velocity - lr * gradient
    parameter_update = updated_velocity
    return updated_velocity, parameter_update

