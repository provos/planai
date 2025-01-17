"""
Debug helper utilities for development and troubleshooting, particularly useful for multi-threaded applications.

This module provides debugging tools that help identify potential bugs and trace execution flow
in complex applications. The logging decorators are especially valuable when debugging issues
that arise from concurrent execution and race conditions in multi-threaded environments.

The utilities help track:
- Function call order and timing
- Caller context and source location
- Argument values passed between functions
- Execution flow across different threads

Usage:
    @log_function_call
    def my_function(arg1, arg2):
        # Function implementation
        pass
"""

import functools
import inspect
import logging


def log_function_call(func):
    """
    Decorator that logs the caller's information, function name, and arguments.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get caller information
        frame = inspect.currentframe().f_back
        caller_name = "Unknown"  # Default if caller information is unavailable
        caller_line = 0

        if frame:
            caller_name = (
                frame.f_code.co_name
                if frame.f_code.co_name != "<module>"
                else "Module-level"
            )
            caller_line = frame.f_lineno

        # Get function name and arguments
        func_name = func.__name__
        arg_names = inspect.getfullargspec(func).args
        all_args = dict(zip(arg_names, args))
        all_args.update(kwargs)

        # Log the call information
        logging.info(
            f"Caller: {caller_name} (line {caller_line}) -> Function: {func_name}, Arguments: {all_args}"
        )

        # Call the original function
        return func(*args, **kwargs)

    return wrapper
