"""Initializes constraints function for Optuna study."""

import numpy as np

def init_constraint_function(metric_name, bounds, storage_key='params'):
    """Initializes constraints function for Optuna study.

    The constraints function returns a sequence of numbers smaller than or equal
    to 0 if the constraints are satisfied.

    Args:
        metric_name (str): Name of the metric to be constrained, or an index if
            values are to be constrained. (e.g., 0 for the first objective)
        bounds (tuple, list, number or callable): Bounds for the metric.
            If tuple or list, it should be (min, max).
            If a single number, it is treated as an upper bound.
            If callable, it should be a function that takes the metric value and returns
            a list of constraint values.
        storage_key (str): Where to find the metric in the trial object.
            Options are 'values', 'user_attrs', or 'params'. Default is 'params'.
    """
    def constraint_function(trial):
        if storage_key == 'values':
            value = trial.values[metric_name]
        elif storage_key == 'user_attrs':
            value = trial.user_attrs[metric_name]
        elif storage_key == 'params':
            value = trial.params[metric_name]
        else:
            raise ValueError("storage_key should be one of 'values', 'user_attrs', or 'params'.")

        if isinstance(bounds, (tuple, list)):
            assert len(bounds) == 2, "Bounds should be a tuple or list of length 2."
            lower, upper = bounds
            if lower is not None and upper is not None:
                return [lower - value,  value - upper]
            elif lower is not None:
                return [lower - value]
            elif upper is not None:
                return [value - upper]
        elif isinstance(bounds, (int, float)):
            # Assume a upper bound if a single number is provided
            return [value - bounds]
        elif callable(bounds):
            return bounds(value)
        else:
            raise ValueError("Bounds should be a tuple, list, single number, or callable.")
    return constraint_function

def initialize_multiple_constraint_functions(metrics, bounds, storage_keys=None):
    """Initializes multiple constraints functions for Optuna study.

    Args:
        metrics (list): List of metric names or indices to be constrained.
        bounds (list): List of bounds for each metric.
        storage_keys (list or None): List of storage keys for each metric.
            If None, all metrics are assumed to be stored in 'params'.

    Returns:
        function: Combined constraints functions.
    """
    if storage_keys is None:
        storage_keys = ['params'] * len(metrics)

    assert len(metrics) == len(bounds) == len(storage_keys), \
        "Metrics, bounds, and storage_keys must have the same length."

    constraint_functions = []
    for metric, bound, storage_key in zip(metrics, bounds, storage_keys):
        constraint_fn = init_constraint_function(metric, bound, storage_key)
        constraint_functions.append(constraint_fn)

    def combined_constraint_function(trial):
        results = []
        for fn in constraint_functions:
            results.extend(fn(trial))
        return results
    return combined_constraint_function
# Example usage:
# constraint_fn = init_constraint_function('accuracy', (0.8, 1.0), storage_key='values')
# multiple_constraints_fn = initialize_multiple_constraint_functions(
#     ['accuracy', 'latency'],
#     [(0.8, 1.0), (None, 100)],
#     ['values', 'user_attrs']
# )
