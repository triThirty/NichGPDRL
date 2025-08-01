import functools
from pathlib import Path
import os
import inspect


def ensure_directory_exists(file_path_arg_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            config = bound_args.arguments["config"]
            file_path = file_path_arg_name.substitute(**config)

            file_path = Path(file_path)

            file_path.parent.mkdir(parents=True, exist_ok=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator
