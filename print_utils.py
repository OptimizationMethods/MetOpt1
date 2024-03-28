from enum import Enum
from numbers import Real
from scipy.optimize import OptimizeResult

import numpy as np


EPS: Real = 1e-12


class Functions(Enum):
	"""
	Enum class for functions string representation
	"""
	SQUARES_SUM: str = "x^2 + y^2"
	LOG_PRODUCT: str = "x^2 * y^2 * log(8x^2 + 3y^2)"


class SelectionMethods(Enum):
	"""
	Enum class for selection methods string representation
	"""
	GOLDEN_SECTION: str = "Золотое сечение"
	TERNARY_SEARCH: str = "Тернарный поиск"
	NON_OPTIMAL_STEP: str = "Неоптимальный шаг"


def print_results(
		x: Real,
		y: Real,
		func_value: Real,
		num_iterations: int,
		execution_time: Real,
		function: Functions,
		selection_method: SelectionMethods
) -> None:
	"""
	Prints the results of the optimization algorithm.
	"""
	print(f"\nФункция {function.value} ({selection_method.value}):")
	print(f"Критерий останова: |delta f| < {EPS}")
	print(f"Число итераций: {num_iterations}")
	print(f"Полученная точка: ({np.round(x, 4)}, {np.round(y, 4)})")
	print(f"Полученное значение функции: {np.round(func_value, 4)}")
	print(f"Время работы: {execution_time:.4f} сек")


def nelder_mead_print(
		result: OptimizeResult,
		function: Functions
) -> None:
	"""
	Prints the results of the Nelder-Mead optimization algorithm.
	"""
	print(f"\nФункция {function.value}:")
	print(f"Полученная точка: ({np.round(result.x[0], 4)}, {np.round(result.x[1], 4)})")
	print(f"Полученное значение функции: {result.fun}")
