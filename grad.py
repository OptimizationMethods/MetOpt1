import time
import numpy as np

def golden_section_search(f, eps):
    phi = (1 + np.sqrt(5)) / 2
    phi2 = 2 - phi
    left = 0
    right = 1
    while abs(right - left) > eps:
        x1 = left + phi2 * (right - left)
        x2 = right - phi2 * (right - left)

        f_new1 = f(x1)
        f_new2 = f(x2)

        if f_new1 < f_new2:
            right = x2
        else:
            left = x1

    return (left + right) / 2

def ternary_search(f, eps):
    left = 0
    right = 1
    while abs(right - left) > eps:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3

        f1 = f(m1)
        f2 = f(m2)

        if f1 < f2:
            right = m2
        else:
            left = m1

    return (left + right) / 2


def gradient_descent(f, grad, x0, y0, selection_method, eps, learning_rate=None, max_iter=1500):
    x_prev = x0
    y_prev = y0
    trajectory = [(x0, y0)]

    iter = 0

    start_time = time.time()
    x = x_prev
    y = y_prev
    for _ in range(max_iter):
        grad_f = grad(x_prev, y_prev)
        if learning_rate is not None:
            alpha = learning_rate
        else:
            f_alpha = lambda alpha: f(x_prev - alpha * grad_f[0], y_prev - alpha * grad_f[1])
            alpha = selection_method(f_alpha, eps)
        x = x_prev - alpha * grad_f[0]
        y = y_prev - alpha * grad_f[1]

        trajectory.append((x, y))

        iter += 1

        if abs(f(x, y) - f(x_prev, y_prev)) < eps:
            break
        x_prev = x
        y_prev = y

    end_time = time.time()
    exec_time = end_time - start_time

    return x, y, iter, exec_time, trajectory
