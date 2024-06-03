import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grad import gradient_descent, golden_section_search, ternary_search
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from print_utils import *

from numbers import Real
from numpy.typing import NDArray


def f1(x: Real, y: Real) -> Real:
    return x ** 2 + y ** 2


def f2(x: Real, y: Real) -> Real:
    return 2 * x ** 2 + 3 * y ** 2 + 4 * x * y + 5 * x + 6 * y


def grad_f1(x: Real, y: Real) -> NDArray[Real]:
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])


def grad_f2(x: Real, y: Real) -> NDArray[Real]:
    df_dx = 4 * x + 4 * y + 5
    df_dy = 6 * y + 4 * x + 6
    return np.array([df_dx, df_dy])


x0: Real = 0.1
y0: Real = 0.1


def find_optimal_learning_rate(f, grad, x0, y0, selection_method, eps, max_iter):
    learning_rates = [2, 1, 0.1, 0.01, 0.001, 0.0001]
    best_learning_rate = learning_rates[0]
    best_execution_time = float('inf')

    for learning_rate in learning_rates:
        _, _, _, exec_time, _ = gradient_descent(f, grad, x0, y0, selection_method, eps, learning_rate, max_iter)
        if exec_time < best_execution_time:
            best_execution_time = exec_time
            best_learning_rate = learning_rate

    return best_learning_rate


best_learning_rate_f1 = find_optimal_learning_rate(f1, grad_f1, x0, y0, golden_section_search, EPS, 1500)
best_learning_rate_f2 = find_optimal_learning_rate(f2, grad_f2, x0, y0, golden_section_search, EPS, 1500)

print(f"Best learning rate for f1: {best_learning_rate_f1}")
print(f"Best learning rate for f2: {best_learning_rate_f2}")


x_neopt1, y_neopt1, num_iterations_neopt1, execution_time_neopt1, trajectory_neopt1 = gradient_descent(f1, grad_f1, x0,
                                                                                                       y0,
                                                                                                       golden_section_search,
                                                                                                       EPS,
                                                                                                       best_learning_rate_f1)

print_results(x_neopt1, y_neopt1,
              f1(x_neopt1, y_neopt1),
              num_iterations_neopt1,
              execution_time_neopt1,
              Functions.SQUARES_SUM,
              SelectionMethods.NON_OPTIMAL_STEP)

x_opt1, y_opt1, num_iterations1, execution_time1, trajectory1 = gradient_descent(f1, grad_f1, x0, y0,
                                                                                 golden_section_search, EPS)

print_results(x_opt1, y_opt1,
              f1(x_opt1, y_opt1),
              num_iterations1,
              execution_time1,
              Functions.SQUARES_SUM,
              SelectionMethods.GOLDEN_SECTION)

x_opt1_ter, y_opt1_ter, num_iterations1_ter, execution_time1_ter, trajectory1_ter = gradient_descent(f1, grad_f1, x0,
                                                                                                     y0, ternary_search,
                                                                                                     EPS)

print_results(x_opt1_ter, y_opt1_ter,
              f1(x_opt1_ter, y_opt1_ter),
              num_iterations1_ter,
              execution_time1_ter,
              Functions.SQUARES_SUM,
              SelectionMethods.TERNARY_SEARCH)

x_neopt2, y_neopt2, num_iterations_neopt2, execution_time_neopt2, trajectory_neopt2 = gradient_descent(f2, grad_f2, x0,
                                                                                                       y0,
                                                                                                       golden_section_search,
                                                                                                       EPS,
                                                                                                       best_learning_rate_f2)
print_results(x_neopt2, y_neopt2,
              f2(x_neopt2, y_neopt2),
              num_iterations_neopt2,
              execution_time_neopt2,
              Functions.ANOTHER_ONE,
              SelectionMethods.NON_OPTIMAL_STEP)

x_opt2, y_opt2, num_iterations2, execution_time2, trajectory2 = gradient_descent(f2, grad_f2, x0, y0,
                                                                                 golden_section_search, EPS)
print_results(x_opt2, y_opt2,
              f2(x_opt2, y_opt2),
              num_iterations2,
              execution_time2,
              Functions.ANOTHER_ONE,
              SelectionMethods.GOLDEN_SECTION)

x_opt2_ter, y_opt2_ter, num_iterations2_ter, execution_time2_ter, trajectory2_ter = gradient_descent(f2, grad_f2, x0,
                                                                                                     y0, ternary_search,
                                                                                                     EPS)
print_results(x_opt2_ter, y_opt2_ter,
              f2(x_opt2_ter, y_opt2_ter),
              num_iterations2_ter,
              execution_time2_ter,
              Functions.ANOTHER_ONE,
              SelectionMethods.TERNARY_SEARCH)


def f1_arr(x: NDArray[Real]) -> Real:
    return f1(x[0], x[1])


def f2_arr(x: NDArray[Real]) -> Real:
    return f2(x[0], x[1])


result_f1 = minimize(f1_arr, np.array([x0, y0]), method='Nelder-Mead', tol=EPS)
result_f2 = minimize(f2_arr, np.array([x0, y0]), method='Nelder-Mead', tol=EPS)

nelder_mead_print(result_f1, Functions.SQUARES_SUM)

nelder_mead_print(result_f2, Functions.ANOTHER_ONE)

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

Z1 = f1(X, Y)
Z2 = f2(X, Y)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='inferno', alpha=0.7)
ax1.set_title('f1: x^2 + y^2')

ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z1, cmap='inferno')
trajectory1 = np.array(trajectory1)
ax2.plot(trajectory1[:, 0], trajectory1[:, 1], 'r--')
ax2.set_title('f1: x^2 + y^2')

plt.show()

fig = plt.figure(figsize=(12, 6))

ax3 = fig.add_subplot(121, projection='3d')
ax3.plot_surface(X, Y, Z2, cmap='inferno', alpha=0.7)
ax3.set_title('f2: 2x^2 + 3y^2 + 4xy + 5x + 6y')

ax4 = fig.add_subplot(122)
ax4.contour(X, Y, Z2, cmap='inferno')
trajectory2 = np.array(trajectory2)
ax4.plot(trajectory2[:, 0], trajectory2[:, 1], 'r--')
ax4.set_title('f2: 2x^2 + 3y^2 + 4xy + 5x + 6y')

plt.show()

df = pd.DataFrame({
    'Function': ['f1', 'f2'],
    'x_opt': [x_opt1, x_opt2],
    'y_opt': [y_opt1, y_opt2],
    'num_iterations': [num_iterations1, num_iterations2],
    'execution_time': [execution_time1, execution_time2]
})

print(df)
