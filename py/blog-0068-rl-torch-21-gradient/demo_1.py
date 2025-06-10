def grad_fx(x: float) -> float:
    # Derivative of f(x) = x^2 is 2x
    return 2 * x


x = 5.0  # Start far from zero
eta = 0.1  # Learning rate

trajectory = [x]
for step in range(20):
    x = x - eta * grad_fx(x)
    trajectory.append(x)
print("Final x:", x)
