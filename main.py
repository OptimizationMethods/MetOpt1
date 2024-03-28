import numpy as np
import pandas as pd
from grad import gradient_descent, golden_section_search, ternary_search
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
eps = 1e-12

def f1(x, y):
    return x ** 2 + y ** 2

def f2(x, y):
    return x ** 2 * y ** 2 * np.log(8 * x**2 + 3 * y**2)

def grad_f1(x, y):
    df_dx = 2*x
    df_dy = 2*y
    return np.array([df_dx, df_dy])

def grad_f2(x, y):
    df_dx = 2 * x * y ** 2 * (np.log(8 * x ** 2 + 3 * y ** 2) + 8 * x ** 2 / (8 * x ** 2 + 3 * y ** 2))
    df_dy = 2 * x ** 2 * y * (np.log(8 * x ** 2 + 3 * y ** 2) + 3 * y ** 2 / (8 * x ** 2 + 3 * y ** 2))
    return np.array([df_dx, df_dy])

x0 = 0.1
y0 = 0.1
learning_rate = 0.1

x_neopt1, y_neopt1, num_iterations_neopt1, execution_time_neopt1, trajectory_neopt1 = gradient_descent(f1, grad_f1, x0, y0, golden_section_search, eps, learning_rate)

print("\nФункция x^2 + y^2 (неоптимальный шаг):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations_neopt1}")
print(f"Полученная точка: ({x_neopt1}, {y_neopt1})")
print(f"Полученное значение функции: {f1(x_neopt1, y_neopt1)}")
print(f"Время работы: {execution_time_neopt1:.4f} сек")

x_opt1, y_opt1, num_iterations1, execution_time1, trajectory1 = gradient_descent(f1, grad_f1, x0, y0, golden_section_search, eps)

print("\nФункция x^2 + y^2 (золотое сечение):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations1}")
print(f"Полученная точка: ({x_opt1}, {y_opt1})")
print(f"Полученное значение функции: {f1(x_opt1, y_opt1)}")
print(f"Время работы: {execution_time1:.4f} сек")

x_opt1_ter, y_opt1_ter, num_iterations1_ter, execution_time1_ter, trajectory1_ter = gradient_descent(f1, grad_f1, x0, y0, ternary_search, eps)

print("\nФункция x^2 + y^2 (тернарный поиск):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations1_ter}")
print(f"Полученная точка: ({x_opt1_ter}, {y_opt1_ter})")
print(f"Полученное значение функции: {f1(x_opt1_ter, y_opt1_ter)}")
print(f"Время работы: {execution_time1_ter:.4f} сек")

x_neopt2, y_neopt2, num_iterations_neopt2, execution_time_neopt2, trajectory_neopt2 = gradient_descent(f2, grad_f2, x0, y0, golden_section_search, eps, learning_rate)

print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2) (неоптимальный шаг):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations_neopt2}")
print(f"Полученная точка: ({x_neopt2}, {y_neopt2})")
print(f"Полученное значение функции: {f2(x_neopt2, y_neopt2)}")
print(f"Время работы: {execution_time_neopt2:.4f} сек")

x_opt2, y_opt2, num_iterations2, execution_time2, trajectory2 = gradient_descent(f2, grad_f2, x0, y0, golden_section_search, eps)

print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2) (золотое сечение):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations2}")
print(f"Полученная точка: ({x_opt2}, {y_opt2})")
print(f"Полученное значение функции: {f2(x_opt2, y_opt2)}")
print(f"Время работы: {execution_time2:.4f} сек")

x_opt2_ter, y_opt2_ter, num_iterations2_ter, execution_time2_ter, trajectory2_ter = gradient_descent(f2, grad_f2, x0, y0, ternary_search, eps)

print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2) (тернарный поиск):")
print(f"Критерий останова: |delta f| < {eps}")
print(f"Число итераций: {num_iterations2_ter}")
print(f"Полученная точка: ({x_opt2_ter}, {y_opt2_ter})")
print(f"Полученное значение функции: {f2(x_opt2_ter, y_opt2_ter)}")
print(f"Время работы: {execution_time2_ter:.4f} сек")


def f1_arr(x):
    return f1(x[0], x[1])

def f2_arr(x):
    return f2(x[0], x[1])

result_f1 = minimize(f1_arr, [x0, y0], method='Nelder-Mead', tol=eps)
result_f2 = minimize(f2_arr, [x0, y0], method='Nelder-Mead', tol=eps)

print("\nФункция x^2 + y^2:")
print(f"Полученная точка: ({result_f1.x[0]}, {result_f1.x[1]})")
print(f"Полученное значение функции: {f2(result_f1.x[0], result_f1.x[1])}")


print("\nФункция x^2 * y^2 * log(8x^2 + 3y^2):")
print(f"Полученная точка: ({result_f2.x[0]}, {result_f2.x[1]})")
print(f"Полученное значение функции: {f2(result_f2.x[0], result_f2.x[1])}\n")


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
ax3.set_title('f2: x^2 * y^2 * log(8x^2 + 3y^2)')


ax4 = fig.add_subplot(122)
ax4.contour(X, Y, Z2, cmap='inferno')
trajectory2 = np.array(trajectory2)
ax4.plot(trajectory2[:, 0], trajectory2[:, 1], 'r--')
ax4.set_title('f2: x^2 * y^2 * log(8x^2 + 3y^2)')

plt.show()


df = pd.DataFrame({
    'Function': ['f1', 'f2'],
    'x_opt': [x_opt1, x_opt2],
    'y_opt': [y_opt1, y_opt2],
    'num_iterations': [num_iterations1, num_iterations2],
    'execution_time': [execution_time1, execution_time2]
})


print(df)