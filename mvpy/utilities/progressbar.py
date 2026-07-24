import threading
import weakref
import joblib

from typing import Union

def _make_tqdm(backend: str = "std", **kwargs):
    if backend == "notebook":
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm(**kwargs)
    elif backend == "auto":
        from tqdm.auto import tqdm as auto_tqdm
        return auto_tqdm(**kwargs)
    else:
        from tqdm import tqdm as std_tqdm
        return std_tqdm(**kwargs)

class _JoblibTqdmRegistry:
    """Bind each active joblib.Parallel instance to the correct tqdm bar."""

    _lock = threading.RLock()
    _patch_depth = 0
    _bars_by_parallel = weakref.WeakKeyDictionary()
    _active = threading.local()

    _old_parallel_call = None
    _old_print_progress = None

    @classmethod
    def _get_stack(cls):
        stack = getattr(cls._active, "stack", None)

        if stack is None:
            stack = []
            cls._active.stack = stack

        return stack

    @classmethod
    def current_bar(cls):
        stack = cls._get_stack()
        return stack[-1] if stack else None

    @classmethod
    def push_bar(cls, bar):
        cls._get_stack().append(bar)

    @classmethod
    def pop_bar(cls, bar):
        stack = cls._get_stack()

        if not stack:
            return

        if stack[-1] is bar:
            stack.pop()
            return

        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is bar:
                del stack[i]
                return

    @classmethod
    def register_parallel(cls, parallel):
        with cls._lock:
            bar = cls.current_bar()

            if bar is not None:
                cls._bars_by_parallel[parallel] = bar

            return bar

    @classmethod
    def get_bar(cls, parallel):
        with cls._lock:
            return cls._bars_by_parallel.get(parallel)

    @classmethod
    def unregister_bar(cls, bar):
        with cls._lock:
            doomed = [p for p, b in cls._bars_by_parallel.items() if b is bar]

            for parallel in doomed:
                del cls._bars_by_parallel[parallel]

    @classmethod
    def patch(cls):
        with cls._lock:
            if cls._patch_depth == 0:
                cls._old_parallel_call = joblib.Parallel.__call__
                cls._old_print_progress = joblib.Parallel.print_progress
                registry = cls

                def patched_call(self, iterable):
                    registry.register_parallel(self)
                    return registry._old_parallel_call(self, iterable)

                def patched_print_progress(self):
                    bar = registry.get_bar(self)

                    if bar is None:
                        return registry._old_print_progress(self)

                    delta = self.n_completed_tasks - bar.n

                    if delta > 0:
                        bar.update(delta)

                joblib.Parallel.__call__ = patched_call
                joblib.Parallel.print_progress = patched_print_progress

            cls._patch_depth += 1

    @classmethod
    def unpatch(cls):
        with cls._lock:
            cls._patch_depth -= 1

            if cls._patch_depth == 0:
                joblib.Parallel.__call__ = cls._old_parallel_call
                joblib.Parallel.print_progress = cls._old_print_progress
                cls._old_parallel_call = None
                cls._old_print_progress = None
                cls._bars_by_parallel = weakref.WeakKeyDictionary()


class _TqdmJoblibContext:
    """Context manager binding a tqdm bar to joblib progress updates."""

    def __init__(self, tqdm_object):
        self.tqdm_object = tqdm_object

    def __enter__(self):
        _JoblibTqdmRegistry.patch()
        _JoblibTqdmRegistry.push_bar(self.tqdm_object)
        return self.tqdm_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        _JoblibTqdmRegistry.pop_bar(self.tqdm_object)
        _JoblibTqdmRegistry.unregister_bar(self.tqdm_object)
        _JoblibTqdmRegistry.unpatch()
        self.tqdm_object.close()

class Progressbar:
    """Simple class for progress bars that can be enabled or disabled.

    Parameters
    ----------
    enabled : bool | int, default=True
        Either whether to enable the progressbar or, if int, at which
        position to place it for tqdm.
    backend : {"std", "auto", "notebook"}, default="std"
        Backend used for tqdm. Use "std" for robust text-based bars,
        including in Jupyter notebooks. Use "notebook" only if widget
        support is working properly.
    **kwargs : Any
        Additional arguments for tqdm.
    """

    def __new__(
        cls,
        enabled: Union[bool, int] = True,
        backend: str = "std",
        **kwargs
    ):
        """Instantiate the progressbar, either as tqdm context or dummy."""

        if "position" not in kwargs:
            kwargs["position"] = max(int(enabled) - 1, 0)

        if "leave" not in kwargs:
            kwargs["leave"] = kwargs["position"] == 0

        if "dynamic_ncols" not in kwargs:
            kwargs["dynamic_ncols"] = True

        if enabled:
            return _TqdmJoblibContext(_make_tqdm(backend=backend, **kwargs))

        return super().__new__(cls)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass